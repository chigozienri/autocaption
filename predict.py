# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import shutil
import tempfile

from cog import BasePredictor, Input, Path

import autocaption

# if __name__ == '__main__':
#   def Input(default=None, **kwargs):
#     return default


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = autocaption.load_model()

    def predict(
        self,
        video_file: Path = Input(description="Video file"),
        subs_position: str = Input(
            description="Subtitles position",
            choices=["bottom75", "center", "top", "bottom", "left", "right"],
            default="bottom75",
        ),
        color: str = Input(description="Caption color", default="white"),
        v_type: str = Input(
            description="Video ratio",
            choices=["9x16", "other aspect ratio"],
            default="other aspect ratio",
        ),
        highlight_color: str = Input(description="Highlight color", default="yellow"),
        fontsize: float = Input(description="Fontsize", default=7.0),
        MaxChars: int = Input(
            description="Max Characters space for subtitles", default=20
        ),
        opacity: float = Input(
            description="Opacity for the subtitles background", default=0.0
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        temp_dir = tempfile.mkdtemp()
        extension = os.path.splitext(video_file)[1]
        videofilename = os.path.join(temp_dir, f"input{extension}")
        shutil.copyfile(video_file, videofilename)

        audiofilename = autocaption.create_audio(videofilename)
        wordlevel_info = autocaption.transcribe_audio(self.model, audiofilename)
        outputfile = autocaption.add_subtitle(
            videofilename,
            audiofilename,
            v_type,
            subs_position,
            highlight_color,
            fontsize,
            opacity,
            MaxChars,
            color,
            wordlevel_info,
        )
        return Path(outputfile)


# if __name__ == "__main__":
#     p = Predictor()
#     p.setup()
#     path = p.predict(video_file='testsubs.mp4')
#     print(path)
