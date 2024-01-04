import time
import uvicorn
import logging
import traceback
import librosa
import edge_tts
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from src.rmvpe import RMVPE
from model_loader import ModelLoader

logger = logging.getLogger(__name__)

app = FastAPI()
model_loader = ModelLoader()
gpu_config = model_loader.config
hubert_model = model_loader.load_hubert()
rmvpe_model = RMVPE("rmvpe.pt", gpu_config.is_half, gpu_config.device)


@app.post("/tts")
async def tts_api(request_data: dict):
    model_name = request_data.get("model_name")
    speed = request_data.get("speed")
    tts_text = request_data.get("tts_text")
    tts_voice = request_data.get("tts_voice")
    f0_up_key = request_data.get("f0_up_key", 0)
    f0_method = request_data.get("f0_method", "rmvpe")
    index_rate = request_data.get("index_rate", 1)
    protect = request_data.get("protect", 0.33)
    filter_radius = request_data.get("filter_radius", 3)
    resample_sr = request_data.get("resample_sr", 0)
    rms_mix_rate = request_data.get("rms_mix_rate", 0.25)

    # temp file
    edge_output_filename = "edge_output.mp3"
    try:
        if len(tts_text) > 280:
            raise HTTPException(
                status_code=400,
                detail=f"Text characters should be at most 280, but got {len(tts_text)} characters.",
            )

        tgt_sr, net_g, vc, version, index_file, if_f0 = model_loader.load(
            model_name
        )

        if speed >= 0:
            speed_str = f"+{speed}%"
        else:
            speed_str = f"{speed}%"
        await edge_tts.Communicate(
            tts_text, "-".join(tts_voice.split("-")[:-1]), rate=speed_str
        ).save(edge_output_filename)

        audio, sr = librosa.load(edge_output_filename, sr=16000, mono=True)
        duration = len(audio) / sr
        if duration >= 20:
            raise HTTPException(
                status_code=400,
                detail=f"Audio should be less than 20 seconds, but got {duration}s.",
            )

        f0_up_key = int(f0_up_key)
        if f0_method == "rmvpe":
            vc.model_rmvpe = rmvpe_model

        times = [0, 0, 0]
        audio_opt = vc.pipeline(
            hubert_model,
            net_g,
            0,
            audio,
            edge_output_filename,
            times,
            f0_up_key,
            f0_method,
            index_file,
            index_rate,
            if_f0,
            filter_radius,
            tgt_sr,
            resample_sr,
            rms_mix_rate,
            version,
            protect,
            None,
        )

        sf.write("output.wav", audio_opt, tgt_sr, format="WAV")

        if tgt_sr != resample_sr >= 16000:
            tgt_sr = resample_sr

        return FileResponse(
            "output.wav",
            headers={"Content-Disposition": "attachment; filename=output.wav"},
        )

    except EOFError:
        info = "It seems that the edge-tts output is not valid. This may occur when the input text and the speaker do not match. For example, maybe you entered Japanese (without alphabets) text but chose a non-Japanese speaker?"
        raise HTTPException(status_code=400, detail=info)
    except Exception as e:
        info = str(e)
        logger.warning(traceback.format_exc())
        raise HTTPException(status_code=500, detail=info)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5021)