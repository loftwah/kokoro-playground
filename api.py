from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, Response, HTMLResponse
import io
from kokoro import KPipeline
import soundfile as sf
from pydub import AudioSegment
import asyncio
from typing import AsyncGenerator
from openai import AsyncOpenAI

app = FastAPI()

# Initialize pipelines for different languages
pipelines = {
    'a': KPipeline(lang_code='a'),  # American English
    'b': KPipeline(lang_code='b'),  # British English
    'f': KPipeline(lang_code='f'),  # French
    'h': KPipeline(lang_code='h'),  # Hindi
}

async def generate_audio_chunks(text: str, voice: str) -> AsyncGenerator[bytes, None]:
    """Generate audio in chunks as it's being processed"""
    # Get the appropriate pipeline based on voice prefix
    lang_code = voice[0]
    pipeline = pipelines[lang_code]
    
    # Split text into smaller chunks for processing
    chunk_size = 200  # characters
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    
    for chunk in chunks:
        # Generate audio for this chunk
        generator = pipeline(chunk, voice=voice)
        for _, _, audio in generator:
            # Convert numpy array to WAV bytes
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, audio, 24000, format='wav')
            wav_buffer.seek(0)
            
            # Convert to MP3 and yield
            audio_segment = AudioSegment.from_wav(wav_buffer)
            chunk_buffer = io.BytesIO()
            audio_segment.export(chunk_buffer, format='mp3')
            
            # Yield this chunk
            yield chunk_buffer.getvalue()
            
            # Small delay to prevent overwhelming the client
            await asyncio.sleep(0.1)

@app.get("/tts")
async def text_to_speech(
    text: str,
    voice: str = "af_heart",
    format: str = "mp3"
):
    try:
        # Validate format
        if format not in ['mp3', 'wav', 'ogg', 'aac', 'opus']:
            raise HTTPException(status_code=400, detail="Unsupported audio format")

        # Get the appropriate pipeline based on voice prefix
        lang_code = voice[0]
        if lang_code not in pipelines:
            raise HTTPException(status_code=400, detail=f"Unsupported language code: {lang_code}")
        
        pipeline = pipelines[lang_code]

        # Generate audio
        generator = pipeline(text, voice=voice)
        audio = None
        for _, _, audio_chunk in generator:
            if audio is None:
                audio = audio_chunk
            else:
                audio = np.concatenate([audio, audio_chunk])

        # Convert to bytes
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio, 24000, format='wav')
        wav_buffer.seek(0)

        # Convert to desired format
        audio_segment = AudioSegment.from_wav(wav_buffer)
        output_buffer = io.BytesIO()

        # Format-specific export settings
        format_settings = {
            'mp3': {'format': 'mp3', 'bitrate': '192k'},
            'wav': {'format': 'wav'},
            'ogg': {'format': 'ogg', 'codec': 'libvorbis', 'bitrate': '192k'},
            'aac': {'format': 'm4a', 'codec': 'aac', 'bitrate': '192k'},
            'opus': {'format': 'opus', 'codec': 'libopus', 'bitrate': '160k'}
        }

        # Export to buffer
        audio_segment.export(output_buffer, **format_settings[format])
        output_buffer.seek(0)

        # Set content type based on format
        content_types = {
            'mp3': 'audio/mpeg',
            'wav': 'audio/wav',
            'ogg': 'audio/ogg',
            'aac': 'audio/aac',
            'opus': 'audio/opus',
        }

        return StreamingResponse(
            output_buffer,
            media_type=content_types[format],
            headers={
                'Content-Disposition': f'attachment; filename="audio.{format}"'
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tts/stream")
async def text_to_speech_stream(
    text: str,
    voice: str = "af_heart",
):
    """Stream audio chunks as they're generated"""
    try:
        # Get the appropriate pipeline based on voice prefix
        lang_code = voice[0]
        if lang_code not in pipelines:
            raise HTTPException(status_code=400, detail=f"Unsupported language code: {lang_code}")
        
        pipeline = pipelines[lang_code]

        return StreamingResponse(
            generate_audio_chunks(text, voice),
            media_type='audio/mpeg',
            headers={
                'Content-Disposition': 'attachment; filename="audio.mp3"'
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Alternative approach: Stream the complete file in chunks
async def iter_file(file_like, chunk_size: int = 1024 * 1024):
    """Stream a file-like object in chunks"""
    while True:
        chunk = file_like.read(chunk_size)
        if not chunk:
            break
        yield chunk

@app.get("/tts/stream-file")
async def text_to_speech_stream_file(
    text: str,
    voice: str = "af_heart",
    format: str = "mp3"
):
    """Generate complete audio first, then stream the file"""
    try:
        # Get the appropriate pipeline based on voice prefix
        lang_code = voice[0]
        if lang_code not in pipelines:
            raise HTTPException(status_code=400, detail=f"Unsupported language code: {lang_code}")
        
        pipeline = pipelines[lang_code]

        # Generate complete audio
        generator = pipeline(text, voice=voice)
        audio = None
        for _, _, audio_chunk in generator:
            if audio is None:
                audio = audio_chunk
            else:
                audio = np.concatenate([audio, audio_chunk])

        # Convert to bytes
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio, 24000, format='wav')
        wav_buffer.seek(0)

        # Convert to desired format
        audio_segment = AudioSegment.from_wav(wav_buffer)
        output_buffer = io.BytesIO()
        audio_segment.export(output_buffer, format=format)
        output_buffer.seek(0)

        content_types = {
            'mp3': 'audio/mpeg',
            'wav': 'audio/wav',
            'ogg': 'audio/ogg',
            'aac': 'audio/aac',
            'opus': 'audio/opus',
        }

        return StreamingResponse(
            iter_file(output_buffer),
            media_type=content_types[format],
            headers={
                'Content-Disposition': f'attachment; filename="audio.{format}"'
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/voices")
async def list_voices():
    """List all available voices"""
    return {
        "voices": [
            {"id": "af_heart", "name": "Heart (Default)"},
            {"id": "af_bella", "name": "Bella (American)"},
            {"id": "af_sarah", "name": "Sarah (American)"},
            {"id": "af_nicole", "name": "Nicole (American)"},
            {"id": "af_sky", "name": "Sky (American)"},
            {"id": "am_adam", "name": "Adam (American)"},
            {"id": "am_michael", "name": "Michael (American)"},
            {"id": "bf_emma", "name": "Emma (British)"},
            {"id": "bf_isabella", "name": "Isabella (British)"},
            {"id": "bm_george", "name": "George (British)"},
            {"id": "bm_lewis", "name": "Lewis (British)"},
            {"id": "ff_siwis", "name": "SIWIS (French)"},
            {"id": "hf_alpha", "name": "Alpha (Hindi)"},
            {"id": "hf_beta", "name": "Beta (Hindi)"}
        ]
    }

# Add this new endpoint
@app.get("/chat", response_class=HTMLResponse)
async def chat_interface():
    """Simple chat interface that speaks responses"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Speaking Chat</title>
        <style>
            #chat-container { margin: 20px; }
            #messages { margin: 20px 0; }
            .message { margin: 10px 0; }
            .user { color: blue; }
            .assistant { color: green; }
        </style>
    </head>
    <body>
        <div id="chat-container">
            <div id="messages"></div>
            <input type="text" id="userInput" style="width: 80%">
            <button onclick="sendMessage()">Send</button>
        </div>

        <script>
            let audioQueue = [];
            let isPlaying = false;

            async function sendMessage() {
                const input = document.getElementById('userInput');
                const messages = document.getElementById('messages');
                
                // Add user message to display
                const userText = input.value;
                messages.innerHTML += `<div class="message user">You: ${userText}</div>`;
                
                // Get LLM response
                const response = await fetch(`/chat/respond?message=${encodeURIComponent(userText)}`);
                const data = await response.json();
                
                // Add assistant message to display
                messages.innerHTML += `<div class="message assistant">Assistant: ${data.text}</div>`;
                
                // Queue the audio
                audioQueue.push(data.text);
                playNextInQueue();
                
                input.value = '';
            }

            async function playNextInQueue() {
                if (isPlaying || audioQueue.length === 0) return;
                
                isPlaying = true;
                const text = audioQueue.shift();
                
                try {
                    const response = await fetch(`/tts/stream?text=${encodeURIComponent(text)}&voice=af_heart`);
                    const audio = new Audio(URL.createObjectURL(await response.blob()));
                    
                    audio.onended = () => {
                        isPlaying = false;
                        playNextInQueue();
                    };
                    
                    audio.play();
                } catch (error) {
                    console.error('Error playing audio:', error);
                    isPlaying = false;
                    playNextInQueue();
                }
            }
        </script>
    </body>
    </html>
    """

@app.get("/chat/respond")
async def chat_respond(message: str):
    """Get LLM response for a message"""
    try:
        # Initialize your LLM client (example using OpenAI)
        client = AsyncOpenAI()
        
        # Get response from LLM
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Keep responses brief and conversational."},
                {"role": "user", "content": message}
            ]
        )
        
        response_text = response.choices[0].message.content
        
        return {
            "text": response_text
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000) 