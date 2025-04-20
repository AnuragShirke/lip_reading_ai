import os
import sys

def check_model_paths():
    """Check if model paths exist and are accessible"""
    model_paths = [
        '/app/Lip_Reading_Using_Deep_Learning/models - checkpoint 96',
        '../Lip_Reading_Using_Deep_Learning/models - checkpoint 96',
        './Lip_Reading_Using_Deep_Learning/models - checkpoint 96',
        '/app/Lip_Reading_Using_Deep_Learning/models - checkpoint 50',
        './Lip_Reading_Using_Deep_Learning/models - checkpoint 50',
        '/app/Lip_Reading_Using_Deep_Learning/models - checkpoint 50/models',
        './Lip_Reading_Using_Deep_Learning/models - checkpoint 50/models'
    ]

    print("Checking model paths...")
    for path in model_paths:
        print(f"Checking: {path}")
        if os.path.exists(path):
            print(f"  ✓ Path exists")
            print(f"  Contents: {os.listdir(path)}")

            # Check for checkpoint file
            checkpoint_path = os.path.join(path, 'checkpoint')
            if os.path.exists(checkpoint_path):
                print(f"  ✓ Checkpoint file exists")
                with open(checkpoint_path, 'r') as f:
                    content = f.read()
                    print(f"  Checkpoint content: {content}")

                # Check for related files
                checkpoint_data = os.path.join(path, 'checkpoint.data-00000-of-00001')
                checkpoint_index = os.path.join(path, 'checkpoint.index')

                if os.path.exists(checkpoint_data):
                    print(f"  ✓ Checkpoint data file exists: {os.path.getsize(checkpoint_data)} bytes")
                else:
                    print(f"  ✗ Checkpoint data file does not exist")

                if os.path.exists(checkpoint_index):
                    print(f"  ✓ Checkpoint index file exists: {os.path.getsize(checkpoint_index)} bytes")
                else:
                    print(f"  ✗ Checkpoint index file does not exist")
            else:
                print(f"  ✗ Checkpoint file does not exist")
        else:
            print(f"  ✗ Path does not exist")

    # Check environment variables
    print("\nChecking environment variables...")
    model_dir = os.environ.get('MODEL_DIR')
    model_path = os.environ.get('MODEL_PATH')
    use_dummy = os.environ.get('USE_DUMMY_MODEL')

    print(f"MODEL_DIR: {model_dir}")
    print(f"MODEL_PATH: {model_path}")
    print(f"USE_DUMMY_MODEL: {use_dummy}")

    if model_dir and os.path.exists(model_dir):
        print(f"MODEL_DIR exists. Contents: {os.listdir(model_dir)}")

    if model_path and os.path.exists(model_path):
        print(f"MODEL_PATH exists.")
        with open(model_path, 'r') as f:
            print(f"MODEL_PATH content: {f.read()}")

    # Check data directory
    data_paths = [
        '/app/Lip_Reading_Using_Deep_Learning/data',
        './Lip_Reading_Using_Deep_Learning/data'
    ]

    print("\nChecking data paths...")
    for path in data_paths:
        print(f"Checking: {path}")
        if os.path.exists(path):
            print(f"  ✓ Path exists")
            print(f"  Contents: {os.listdir(path)}")

            # Check for s1 directory
            s1_path = os.path.join(path, 's1')
            if os.path.exists(s1_path):
                print(f"  ✓ s1 directory exists")
                print(f"  s1 contents (first 5 files): {os.listdir(s1_path)[:5]}")

                # Check for specific test file
                test_file = os.path.join(s1_path, 'bbaf2n.mpg')
                if os.path.exists(test_file):
                    print(f"  ✓ Test file bbaf2n.mpg exists: {os.path.getsize(test_file)} bytes")
                else:
                    print(f"  ✗ Test file bbaf2n.mpg does not exist")
            else:
                print(f"  ✗ s1 directory does not exist")

            # Check for alignments directory
            alignments_path = os.path.join(path, 'alignments')
            if os.path.exists(alignments_path):
                print(f"  ✓ alignments directory exists")
                s1_align_path = os.path.join(alignments_path, 's1')
                if os.path.exists(s1_align_path):
                    print(f"  ✓ alignments/s1 directory exists")
                    print(f"  alignments/s1 contents (first 5 files): {os.listdir(s1_align_path)[:5]}")

                    # Check for specific test file
                    test_align_file = os.path.join(s1_align_path, 'bbaf2n.align')
                    if os.path.exists(test_align_file):
                        print(f"  ✓ Test file bbaf2n.align exists: {os.path.getsize(test_align_file)} bytes")
                        with open(test_align_file, 'r') as f:
                            print(f"  bbaf2n.align content: {f.read()[:100]}...")
                    else:
                        print(f"  ✗ Test file bbaf2n.align does not exist")
                else:
                    print(f"  ✗ alignments/s1 directory does not exist")
            else:
                print(f"  ✗ alignments directory does not exist")
        else:
            print(f"  ✗ Path does not exist")

if __name__ == "__main__":
    check_model_paths()
