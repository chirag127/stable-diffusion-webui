import gradio as gr
import numpy as np
from PIL import Image

from modules import gfpgan_model, scripts_postprocessing, ui_components


class ScriptPostprocessingGfpGan(scripts_postprocessing.ScriptPostprocessing):
    name = "GFPGAN"
    order = 2000

    def ui(self):
        with ui_components.InputAccordion(False, label="GFPGAN") as enable:
            gfpgan_visibility = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                step=0.001,
                label="Visibility",
                value=1.0,
                elem_id="extras_gfpgan_visibility",
            )

        return {
            "enable": enable,
            "gfpgan_visibility": gfpgan_visibility,
        }

    def process(
        self, pp: scripts_postprocessing.PostprocessedImage, enable, gfpgan_visibility
    ):
        if gfpgan_visibility == 0 or not enable:
            return

        restored_img = gfpgan_model.gfpgan_fix_faces(
            np.array(pp.image.convert("RGB"), dtype=np.uint8)
        )
        res = Image.fromarray(restored_img)

        if gfpgan_visibility < 1.0:
            res = Image.blend(pp.image, res, gfpgan_visibility)

        pp.image = res
        pp.info["GFPGAN visibility"] = round(gfpgan_visibility, 3)
