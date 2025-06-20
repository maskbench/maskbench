import requests
import zipfile
import os
import argparse
import json


api_url = "http://maskanyone:8000"

def send_post_request(api_url, input_video, output_video, hiding_strategy, overlay_strategy):
    with open(input_video, 'rb') as video_file:
        files = {'video': (os.path.basename(input_video), video_file, 'video/mp4')}
        data = {
            'options': json.dumps({
                'hiding_strategy': hiding_strategy,
                'overlay_strategy': overlay_strategy
            })
        }
        response = requests.post(f"{api_url}/mask-video", files=files, data=data)
        if response.status_code == 200:
            with open(output_video, 'wb') as output_file:
                output_file.write(response.content)
            print(f"Video processed successfully and saved to {output_video}")
        else:
            print(f"Failed to process video: {response.status_code} - {response.text}")

   
def main():
    parser = argparse.ArgumentParser(description="MaskAnyone CLI for video processing")
    parser.add_argument('--input-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--hiding-strategy', type=str, required=True)
    parser.add_argument('--overlay-strategy', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    if os.path.isfile(args.input_path) and args.input_path.endswith(".mp4"):
        output_file = os.path.join(
            args.output_path,
            f"{os.path.splitext(os.path.basename(args.input_path))[0]}_processed.mp4"
        )
        send_post_request(api_url, args.input_path, output_file, args.hiding_strategy, args.overlay_strategy)

    elif os.path.isdir(args.input_path):
        for file_name in os.listdir(args.input_path):
            if file_name.endswith('.mp4'):
                input_video = os.path.join(args.input_path, file_name)
                output_video = os.path.join(args.output_path, f"{os.path.splitext(file_name)[0]}_processed.mp4")
                send_post_request(api_url, input_video, output_video, args.hiding_strategy, args.overlay_strategy)
    else:
        print("Invalid input path")

if __name__ == "__main__":
    main()