"""API client for testing deepfake audio detection API."""
import os
import sys
import time
import argparse

import requests


def test_health(base_url: str):
    """Test health endpoint.
    
    Args:
        base_url: Base URL of API
    """
    print("\n" + "="*70)
    print("TEST 1: HEALTH CHECK")
    print("="*70)
    
    try:
        response = requests.get(f"{base_url}/health")
        response.raise_for_status()
        
        data = response.json()
        print(f"\n✅ Health check successful:")
        print(f"   Status: {data['status']}")
        print(f"   Model loaded: {data['model_loaded']}")
        print(f"   Device: {data['device']}")
        
        return True
    except Exception as e:
        print(f"\n❌ Health check failed: {e}")
        return False


def test_info(base_url: str):
    """Test info endpoint.
    
    Args:
        base_url: Base URL of API
    """
    print("\n" + "="*70)
    print("TEST 2: MODEL INFO")
    print("="*70)
    
    try:
        response = requests.get(f"{base_url}/info")
        response.raise_for_status()
        
        data = response.json()
        print(f"\n✅ Model info retrieved:")
        print(f"   Model path: {data['model_path']}")
        print(f"   Feature type: {data['feature_type']}")
        print(f"   Sample rate: {data['sample_rate']}Hz")
        print(f"   Device: {data['device']}")
        print(f"   Max file size: {data['max_file_size_mb']}MB")
        print(f"   Allowed extensions: {', '.join(data['allowed_extensions'])}")
        
        return True
    except Exception as e:
        print(f"\n❌ Info request failed: {e}")
        return False


def test_prediction(base_url: str, audio_file: str):
    """Test prediction endpoint.
    
    Args:
        base_url: Base URL of API
        audio_file: Path to audio file
    """
    print("\n" + "="*70)
    print("TEST 3: PREDICTION")
    print("="*70)
    print(f"\nAudio file: {audio_file}")
    
    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        return False
    
    try:
        # Upload file
        print("\nUploading file...")
        with open(audio_file, 'rb') as f:
            files = {'file': (os.path.basename(audio_file), f, 'audio/wav')}
            
            start_time = time.time()
            response = requests.post(f"{base_url}/predict", files=files)
            elapsed = time.time() - start_time
            
            response.raise_for_status()
        
        data = response.json()
        
        print(f"\n✅ Prediction successful (took {elapsed:.2f}s):")
        print(f"\n   Prediction: {data['prediction'].upper()}")
        print(f"   Confidence: {data['confidence'] * 100:.2f}%")
        print(f"\n   Probabilities:")
        print(f"      Real: {data['probabilities']['real'] * 100:.2f}%")
        print(f"      Fake: {data['probabilities']['fake'] * 100:.2f}%")
        print(f"\n   Metadata:")
        print(f"      Filename: {data['metadata']['filename']}")
        print(f"      Duration: {data['metadata']['duration_seconds']:.2f}s")
        print(f"      Sample rate: {data['metadata']['sample_rate']}Hz")
        print(f"      Feature type: {data['metadata']['feature_type']}")
        print(f"      Feature shape: {data['metadata']['feature_shape']}")
        
        return True
    except requests.exceptions.HTTPError as e:
        print(f"\n❌ Prediction failed (HTTP {e.response.status_code}):")
        try:
            error_data = e.response.json()
            print(f"   Error: {error_data.get('error', 'Unknown')}")
            print(f"   Message: {error_data.get('message', 'No message')}")
        except:
            print(f"   {e}")
        return False
    except Exception as e:
        print(f"\n❌ Prediction failed: {e}")
        return False


def test_batch_prediction(base_url: str, audio_files: list):
    """Test batch predictions.
    
    Args:
        base_url: Base URL of API
        audio_files: List of audio file paths
    """
    print("\n" + "="*70)
    print("TEST 4: BATCH PREDICTIONS")
    print("="*70)
    
    results = []
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\nFile {i}/{len(audio_files)}: {audio_file}")
        
        if not os.path.exists(audio_file):
            print(f"   ⚠️  File not found, skipping")
            continue
        
        try:
            with open(audio_file, 'rb') as f:
                files = {'file': (os.path.basename(audio_file), f, 'audio/wav')}
                response = requests.post(f"{base_url}/predict", files=files)
                response.raise_for_status()
            
            data = response.json()
            results.append(data)
            
            print(f"   ✓ {data['prediction'].upper()} ({data['confidence']*100:.2f}%)")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    if results:
        print(f"\n✅ Batch prediction complete: {len(results)}/{len(audio_files)} successful")
    
    return len(results) > 0


def test_error_handling(base_url: str):
    """Test error handling.
    
    Args:
        base_url: Base URL of API
    """
    print("\n" + "="*70)
    print("TEST 5: ERROR HANDLING")
    print("="*70)
    
    # Test 1: No file
    print("\n1. Testing with no file...")
    try:
        response = requests.post(f"{base_url}/predict")
        if response.status_code == 400:
            print(f"   ✓ Correctly rejected (400): {response.json()['error']}")
        else:
            print(f"   ⚠️  Unexpected status: {response.status_code}")
    except Exception as e:
        print(f"   ⚠️  Error: {e}")
    
    # Test 2: Invalid file type
    print("\n2. Testing with invalid file type...")
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b"This is not an audio file")
            temp_path = f.name
        
        try:
            with open(temp_path, 'rb') as f:
                files = {'file': ('test.txt', f, 'text/plain')}
                response = requests.post(f"{base_url}/predict", files=files)
            
            if response.status_code == 400:
                print(f"   ✓ Correctly rejected (400): {response.json()['error']}")
            else:
                print(f"   ⚠️  Unexpected status: {response.status_code}")
        finally:
            os.unlink(temp_path)
    except Exception as e:
        print(f"   ⚠️  Error: {e}")
    
    print("\n✅ Error handling tests complete")


def main():
    """Run API client tests."""
    parser = argparse.ArgumentParser(description='Test Deepfake Detection API')
    parser.add_argument('--url', type=str, default='http://localhost:5000',
                        help='API base URL')
    parser.add_argument('--audio', type=str, nargs='+',
                        default=['test_audio/test_audio_1.wav'],
                        help='Audio files to test')
    parser.add_argument('--skip-errors', action='store_true',
                        help='Skip error handling tests')
    
    args = parser.parse_args()
    
    print("="*70)
    print("API CLIENT TEST SUITE")
    print("="*70)
    print(f"\nAPI URL: {args.url}")
    print(f"Audio files: {len(args.audio)}")
    
    # Check if server is running
    print("\nChecking if API server is running...")
    try:
        response = requests.get(args.url, timeout=2)
        print("✅ Server is running")
    except requests.exceptions.ConnectionError:
        print("\n❌ Cannot connect to API server!")
        print(f"\nPlease start the server first:")
        print(f"   python src/deployment/api.py --model models/test_model_scripted.pt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    # Run tests
    results = []
    
    # Test 1: Health
    results.append(('Health Check', test_health(args.url)))
    
    # Test 2: Info
    results.append(('Model Info', test_info(args.url)))
    
    # Test 3: Single prediction
    results.append(('Single Prediction', test_prediction(args.url, args.audio[0])))
    
    # Test 4: Batch predictions
    if len(args.audio) > 1:
        results.append(('Batch Predictions', test_batch_prediction(args.url, args.audio)))
    
    # Test 5: Error handling
    if not args.skip_errors:
        test_error_handling(args.url)
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        sys.exit(1)


if __name__ == '__main__':
    main()
