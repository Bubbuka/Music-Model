

from miditok import REMI, TokenizerConfig
from symusic import Score
from pathlib import Path
import json

config = TokenizerConfig(num_velocities=16, use_chords=True, use_programs=True)
tokenizer = REMI(config)

files_paths = list(Path("/Users/mateoterreni/Desktop/Ai_music_project/Dataset/AC_DC/").glob("**/*.mid"))


tokenizer.train(vocab_size=30000, files_paths=files_paths)
tokenizer.save(Path("./tokenizer.json"))


for midi_path in files_paths:
    original_midi = Score(midi_path)
    tokens = tokenizer(original_midi)
    
    # Do something with the tokens, e.g., save them or use them for training



detokenized_midi = tokenizer(tokens)

# Step 6: Compare the original and detokenized MIDI files
def compare_midi_files(midi1, midi2):
    """Compare two MIDI files and print their differences."""
    print("\nComparing MIDI files...")

    # Compare general information
    print(f"Original MIDI - Number of tracks: {len(midi1.tracks)}")
    print(f"Detokenized MIDI - Number of tracks: {len(midi2.tracks)}")
    print(f"Original MIDI - Ticks per quarter note: {midi1.ticks_per_quarter}")
    print(f"Detokenized MIDI - Ticks per quarter note: {midi2.ticks_per_quarter}")

    # Compare tracks
    for i, (track1, track2) in enumerate(zip(midi1.tracks, midi2.tracks)):
        print(f"\nTrack {i}:")
        print(f"Original MIDI - Number of notes: {len(track1.notes)}")
        print(f"Detokenized MIDI - Number of notes: {len(track2.notes)}")
        print(f"Original MIDI - Number of control changes: {len(track1.controls)}")
        print(f"Detokenized MIDI - Number of control changes: {len(track2.controls)}")

        # Compare notes
        for note1, note2 in zip(track1.notes, track2.notes):
            if note1.pitch != note2.pitch or note1.velocity != note2.velocity or \
               note1.start != note2.start or note1.end != note2.end:
                print(f"  Note mismatch: Original={note1}, Detokenized={note2}")

    # Compare tempo changes
    for tempo1, tempo2 in zip(midi1.tempos, midi2.tempos):
        if tempo1.time != tempo2.time or tempo1.tempo != tempo2.tempo:
            print(f"Tempo mismatch: Original={tempo1}, Detokenized={tempo2}")

    # Compare time signatures
    for ts1, ts2 in zip(midi1.time_signatures, midi2.time_signatures):
        if ts1.time != ts2.time or ts1.numerator != ts2.numerator or ts1.denominator != ts2.denominator:
            print(f"Time signature mismatch: Original={ts1}, Detokenized={ts2}")

    print("\nComparison complete.")

# Step 7: Compare the original and detokenized MIDI files
compare_midi_files(original_midi, detokenized_midi)