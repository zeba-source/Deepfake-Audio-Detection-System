from datasets import load_dataset
from datasets.features import Features, Value

# Load in streaming mode but avoid decoding audio by overriding the 'audio' feature
# to a simple string (path or reference). This prevents torchcodec/FFmpeg requirements.
try:
	# Try to load dataset metadata and a single example without audio decoding.
	# The dataset's parquet schema contains an 'audio' struct with children 'bytes' and 'path'.
	# Provide a matching Features mapping so the datasets library will not attempt to decode
	# the audio as an Audio feature (which triggers torchcodec/FFmpeg).
	audio_struct_features = Features({
		"audio": Features({
			"bytes": Value("binary"),
			"path": Value("string"),
		}),
		# other fields we saw in the schema
		"audio_id": Value("string"),
		"real_or_fake": Value("string"),
	})

	ds = load_dataset(
		"ajaykarthick/wavefake-audio",
		verification_mode='no_checks',
		streaming=True,
		features=audio_struct_features,
	)

	print("Splits:", list(ds.keys()))
	# Get first split name
	split = list(ds.keys())[0]
	# Peek into the dataset's column names and features (no decoding)
	# The library resolves data files on init; examine dataset info where available.
	from datasets import get_dataset_config_names
	print("Columns (sample): attempting to fetch one example's keys without decoding...")

	# Use low-level iterable access but do not trigger audio decoding: fetch the raw arrow table
	# Instead of decoding batch items, we'll request one record's metadata by reading the dataset's files list
	# Use the builder's info if available.
	try:
		# Print dataset info if present
		if hasattr(ds[split], 'info') and ds[split].info is not None:
			info = ds[split].info
			print('Dataset description:', getattr(info, 'description', '')[:300])
			print('Number of rows (if known):', getattr(info.splits, 'num_rows', 'unknown'))
	except Exception:
		pass

	# As a robust fallback, iterate one item but without attempting to decode audio fields.
	# We'll extract only keys using the dataset's column names if available via `column_names`.
	try:
		colnames = ds[split].column_names
		print('Column names:', colnames)
	except Exception:
		print('Could not read column names.')

	# Print up to 5 metadata rows (audio.path, audio_id, real_or_fake) without decoding audio bytes.
	try:
		it = iter(ds[split])
		print('\nFirst up to 5 metadata rows:')
		for i in range(5):
			try:
				row = next(it)
			except StopIteration:
				break
			audio = row.get('audio', None)
			audio_path = None
			try:
				if isinstance(audio, dict):
					audio_path = audio.get('path') or audio.get('source') or None
			except Exception:
				audio_path = None

			print(f"{i+1}. audio_path={repr(audio_path)}, audio_id={repr(row.get('audio_id'))}, real_or_fake={repr(row.get('real_or_fake'))}")
	except Exception as e:
		print('Could not fetch metadata rows:', e)

except Exception as e:
	print('Failed to load dataset metadata:', e)