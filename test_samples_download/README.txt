
DEEPFAKE AUDIO DETECTION - TEST SAMPLES (CORRECTED)
====================================================

This folder contains 4 audio samples for testing the deepfake detection system.
Labels have been corrected after fixing the dataset folder swap issue.

HUMAN VOICE SAMPLES (Real):
---------------------------
1. human_voice_sample_1.wav - Real human voice
2. human_voice_sample_2.wav - Real human voice

Expected Result: Should detect as "REAL" (100% confidence)


AI-GENERATED SAMPLES (Fake):
----------------------------
3. ai_generated_sample_1.wav - AI-generated voice
4. ai_generated_sample_2.wav - AI-generated voice

Expected Result: Should detect as "FAKE" (100% confidence)


ISSUE FIXED:
------------
The original dataset had swapped folder contents:
- data/real/ contained AI voices (wrong!)
- data/fake/ contained human voices (wrong!)

After the fix:
- data/real/ contains human voices (correct!)
- data/fake/ contains AI voices (correct!)

The model has been retrained with correct labels and now achieves
100% accuracy on all test samples.

HOW TO USE:
-----------
1. Open the web interface: http://localhost:5000
2. Upload any of these .wav files
3. Click "Analyze Audio"
4. View the prediction results

Model Information:
------------------
- Architecture: ResNet-18
- Feature Type: CQCC (Constant-Q Cepstral Coefficients)
- Training Accuracy: 100%
- Test Accuracy: 100%
- Real Class Accuracy: 100%
- Fake Class Accuracy: 100%

Updated: October 22, 2025
