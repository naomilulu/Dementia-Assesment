import torch
from pyannote.audio import Pipeline
from pydub import AudioSegment
import os

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="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")

# send pipeline to GPU (when available)
pipeline.to(torch.device("cuda"))

path = "CTTsegment"
path2 = "CTTsegment_diarization"

audiolist = []

for f in os.listdir(path):
    filepath = path + '/' + f
    filepath2 = filepath.replace("CTTsegment", "CTTsegment_diarization").replace(".wav", "")
    if os.path.isfile(filepath) and f.find("wav") != -1:

        # apply pretrained pipeline
        diarization = pipeline(filepath, min_speakers=1, max_speakers=5)

        # # print the result
        # for turn, _, speaker in diarization.itertracks(yield_label=True):
        #     if(speaker == "SPEAKER_00"):
        #         print("PAD: ")
        #     print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")

        # Initialize empty dictionary to store speaker segments
        speaker_segments = {}

        # Iterate over diarization result and store speaker segments
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
            print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
            speaker_segments[speaker].append(turn)
            
        # print(speaker_segments)
        print(filepath2)

        # Save each speaker segment to a separate file
        for speaker, timeline in speaker_segments.items():
            combined_segment = AudioSegment.empty()
            for segment in timeline:
                audio_segment = AudioSegment.from_wav(filepath)[int(segment.start * 1000):int(segment.end * 1000)]
                combined_segment += audio_segment
            idx = speaker.split("SPEAKER_")[-1]
            combined_segment.export(filepath2 + f"_{idx}.wav", format="wav")

        # Combine all speaker files into one
        combined_audio = AudioSegment.empty()
        for speaker, _ in speaker_segments.items():
            idx = speaker.split("SPEAKER_")[-1]
            combined_audio += AudioSegment.from_file(filepath2 + f"_{idx}.wav")

        # Export the combined audio
        combined_audio.export(filepath2 + "_combine.wav", format="wav")