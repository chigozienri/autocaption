# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import json
import os
import shutil
import tempfile
from typing import List

from cog import BasePredictor, Input, Path

import autocaption

if __name__ == "__main__":

    def Input(default=None, **kwargs):
        return default


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = autocaption.load_model()

    def predict(
        self,
        video_file_input: Path = Input(description="Video file"),
        transcript_file_input: Path = Input(
            description="Transcript file, if provided will use this for words rather than whisper.",
            default=None,
        ),
        output_video: bool = Input(
            description="Output video, if true will output the video with subtitles",
            default=True,
        ),
        output_transcript: bool = Input(
            description="Output transcript json, if true will output a transcript file that you can edit and use for the next run in transcript_file_input",
            default=True,
        ),
        subs_position: str = Input(
            description="Subtitles position",
            choices=["bottom75", "center", "top", "bottom", "left", "right"],
            default="bottom75",
        ),
        color: str = Input(description="Caption color", default="white"),
        highlight_color: str = Input(description="Highlight color", default="yellow"),
        fontsize: float = Input(
            description="Font size. 7.0 is good for videos, 4.0 is good for reels",
            default=7.0,
        ),
        MaxChars: int = Input(
            description="Max characters space for subtitles. 20 is good for videos, 10 is good for reels",
            default=20,
        ),
        opacity: float = Input(
            description="Opacity for the subtitles background", default=0.0
        ),
        font: str = Input(
            description="Font",
            default="Poppins/Poppins-ExtraBold.ttf",
            choices=[
                "Poppins/Poppins-Bold.ttf",
                "Poppins/Poppins-BoldItalic.ttf",
                "Poppins/Poppins-ExtraBold.ttf",
                "Poppins/Poppins-ExtraBoldItalic.ttf",
                "Poppins/Poppins-Black.ttf",
                "Poppins/Poppins-BlackItalic.ttf",
                "Atkinson_Hyperlegible/AtkinsonHyperlegible-Bold.ttf",
                "Atkinson_Hyperlegible/AtkinsonHyperlegible-BoldItalic.ttf",
                "M_PLUS_Rounded_1c/MPLUSRounded1c-ExtraBold.ttf",
                "Arial/Arial_Bold.ttf",
                "Arial/Arial_BoldItalic.ttf"
            ],
        ),
        stroke_color: str = Input(description="Stroke color", default="black"),
        stroke_width: float = Input(description="Stroke width", default=2.6),
        kerning: float = Input(description="Kerning for the subtitles", default=-5.0),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        temp_dir = tempfile.mkdtemp()
        extension = os.path.splitext(video_file_input)[1]
        videofilename = os.path.join(temp_dir, f"input{extension}")
        shutil.copyfile(video_file_input, videofilename)
        if transcript_file_input:
            shutil.copyfile(
                transcript_file_input, os.path.join(temp_dir, "transcript_in.json")
            )
            with open(os.path.join(temp_dir, "transcript_in.json")) as f:
                wordlevel_info = json.loads(f.read())
        else:
            audiofilename = autocaption.create_audio(videofilename)
            wordlevel_info = autocaption.transcribe_audio(self.model, audiofilename)
        outputs = []
        if output_video:
            outputfile = autocaption.add_subtitle(
                videofilename,
                "other aspect ratio",  # v_type is unused
                subs_position,
                highlight_color,
                fontsize,
                opacity,
                MaxChars,
                color,
                wordlevel_info,
                font,
                stroke_color,
                stroke_width,
                kerning,
            )
            outputs.append(Path(outputfile))
        if output_transcript:
            transcript_file_output = os.path.join(temp_dir, "transcript_out.json")
            with open(transcript_file_output, "w") as f:
                f.write(json.dumps(wordlevel_info, indent=4))
            outputs.append(Path(transcript_file_output))
        return outputs


if __name__ == "__main__":
    p = Predictor()
    p.setup()
    path = p.predict(video_file_input="kingnobel.mp4", kerning=5)
    print(path)
