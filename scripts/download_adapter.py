"""Download the DemoiréEdit adapter and verify its pinned source identity."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-id', default='umu/DemoireEdit', help='Hugging Face DemoiréEdit model repository')
    parser.add_argument('--revision', default='be28c72d00e8ee3aa01e84c7ad063bdc18c7e604', help='Use a commit hash for a fixed revision')
    parser.add_argument('--output', type=Path, default=Path('weights'))
    args = parser.parse_args()
    from huggingface_hub import hf_hub_download
    from demoire_edit.infer import verify_lora
    path = hf_hub_download(repo_id=args.repo_id, filename='demoireedit.safetensors',
                           revision=args.revision, local_dir=args.output)
    verify_lora(path)
    print('Verified adapter: ' + str(Path(path)))


if __name__ == '__main__':
    main()
