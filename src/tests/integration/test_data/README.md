# Integration Test Data

This directory contains test data files for integration testing.

## Required Files

1. `test_interview.wav`
   - Sample interview audio file
   - Format: WAV
   - Sample rate: 16000 Hz
   - Duration: ~30 seconds
   - Content: Mock interview question and answer

2. `test_cv.pdf`
   - Sample CV/resume
   - Format: PDF
   - Content: Mock candidate information
   - Should include:
     - Technical skills
     - Work experience
     - Education
     - Projects

## File Requirements

### test_interview.wav
- Must be clear speech
- Should contain both interviewer and candidate
- Should include technical discussion
- No background noise
- Mono channel
- 16-bit PCM

### test_cv.pdf
- Standard CV format
- Machine-readable text (not scanned)
- Include realistic technical content
- No personal identifying information

## Usage

These files are used by the integration tests to verify:
1. Audio transcription
2. Document processing
3. Context awareness
4. Real-time analysis

## Creating Test Files

### Audio File
```bash
# Record test audio (Linux)
arecord -f S16_LE -c1 -r16000 -d30 test_interview.wav

# Record test audio (macOS)
sox -d -c 1 -r 16000 test_interview.wav
```

### CV File
- Create using any PDF editor
- Use mock data
- Include relevant technical content
- Save as searchable PDF

## Note
Do not commit real personal data. Use mock data only.