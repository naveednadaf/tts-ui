import streamlit as st
from kokoro import KPipeline
import io
import wave
import numpy as np

def main():
    st.title("Kokoro TTS Web Interface")
    
    # Initialize Kokoro Pipeline
    @st.cache_resource
    def load_tts_pipeline():
        try:
            # Create a pipeline for American English ('a')
            return KPipeline(lang_code="a")  # American English pipeline
        except Exception as e:
            st.error(f"Error initializing Kokoro pipeline: {e}")
            return None
    
    pipeline = load_tts_pipeline()
    
    if pipeline is None:
        st.error("Failed to initialize Kokoro TTS pipeline. Please check your installation.")
        return
    
    # Text input
    text_input = st.text_area("Enter text to convert to speech:", height=150)
    
    # Generation options
    col1, col2 = st.columns(2)
    
    with col1:
        speed = st.slider("Speed", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
        
    with col2:
        # Updated voice selection with actual available voices
        voice_selection = st.selectbox(
            "Voice", 
            [
                "af_bella",      # Female voice
                "af_sarah",      # Female voice
                "af_nova",       # Female voice
                "am_michael",    # Male voice
                "am_liam",       # Male voice
                "am_eric",       # Male voice
                "bm_george",     # Male voice
                "bm_lewis",      # Male voice
                "af_aoede",      # Female voice
                "af_sky",        # Female voice
                "am_puck",       # Male voice
            ]
        )  
    
    # Generate button
    if st.button("Generate Speech"):
        if text_input.strip():
            with st.spinner("Generating audio... This may take a moment as the model downloads if it hasn't been used before."):
                try:
                    # Generate audio using the pipeline
                    # The pipeline yields results, so we need to iterate through them
                    audio_generated = False
                    for result in pipeline(text_input, voice=voice_selection, speed=speed):
                        if result.output is not None:
                            # Extract audio data from the result
                            audio_tensor = result.output.audio
                            
                            # Convert tensor to numpy array
                            if hasattr(audio_tensor, 'cpu'):
                                audio_np = audio_tensor.cpu().numpy()
                            else:
                                audio_np = audio_tensor.numpy()
                            
                            # Normalize audio to prevent clipping
                            if len(audio_np) > 0 and audio_np.max() != 0:
                                audio_np = audio_np / abs(audio_np).max()
                            
                            # Convert to 16-bit PCM
                            audio_pcm = (audio_np * 32767).astype(np.int16)
                            
                            # Create WAV file in memory
                            wav_buffer = io.BytesIO()
                            with wave.open(wav_buffer, 'wb') as wav_file:
                                wav_file.setnchannels(1)  # Mono
                                wav_file.setsampwidth(2)  # 16-bit
                                wav_file.setframerate(22050)  # Sample rate
                                wav_file.writeframes(audio_pcm.tobytes())
                            
                            # Seek to the beginning of the buffer
                            wav_buffer.seek(0)
                            
                            # Play the audio
                            st.audio(wav_buffer, format='audio/wav')
                            st.success(f"Audio generated successfully with voice: {voice_selection}!")
                            audio_generated = True
                            break  # Just get the first result for simplicity
                    
                    if not audio_generated:
                        st.warning("No audio output generated. Check your input text and voice selection.")
                
                except Exception as e:
                    st.error(f"Error generating audio: {str(e)}")
                    st.error("Make sure you have a valid voice selected and internet connection for downloading models if needed.")
        else:
            st.warning("Please enter some text to convert to speech.")

if __name__ == "__main__":
    main()