import { Client, handle_file } from "https://cdn.jsdelivr.net/npm/@gradio/client/dist/index.min.js";

const statusEl = document.getElementById("status");
function setStatus(text, cls = "") {
  statusEl.textContent = text;
  statusEl.className = "status " + cls;
}

const client = await Client.connect(window.location.origin);
console.log("Connected to gradio.Server");

// ---------- Tabs ----------
document.querySelectorAll(".tab").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById("tab-" + btn.dataset.tab).classList.add("active");
  });
});

// ---------- Helpers ----------
const $ = id => document.getElementById(id);

function fillSelect(el, values, selected) {
  if (!el) return;
  const cur = selected ?? el.value;
  el.innerHTML = "";
  (values || []).forEach(v => {
    const o = document.createElement("option");
    o.value = v; o.textContent = v;
    el.appendChild(o);
  });
  if (cur && (values || []).includes(cur)) el.value = cur;
}

// ---------- Load meta ----------
let META = {};
async function loadMeta() {
  const r = await client.predict("/meta", {});
  META = r.data[0];
  fillSelect($("model_file"), META.models);
  fillSelect($("index_file"), META.indexes);
  fillSelect($("audio"), META.audios);
  fillSelect($("vocal_model"), META.vocals_models, "Mel-Roformer by KimberleyJSN");
  fillSelect($("karaoke_model"), META.karaoke_models, "Mel-Roformer Karaoke by aufr33 and viperx");
  fillSelect($("dereverb_model"), META.dereverb_models, "UVR-Deecho-Dereverb");
  fillSelect($("deeecho_model"), META.deeecho_models, "UVR-Deecho-Normal");
  fillSelect($("denoise_model"), META.denoise_models, "Mel-Roformer Denoise Normal by aufr33");
  $("devices").value = META.devices;
  if ($("model_file").value) syncIndex($("model_file").value);
}

async function syncIndex(modelFile) {
  if (!modelFile) return;
  const r = await client.predict("/match_index", { model_file: modelFile });
  if (r.data[0]) $("index_file").value = r.data[0];
}

$("model_file").addEventListener("change", e => syncIndex(e.target.value));

// ---------- Refresh ----------
$("refresh").addEventListener("click", async () => {
  const r = await client.predict("/refresh_lists", {});
  const { models, indexes, audios } = r.data[0];
  fillSelect($("model_file"), models);
  fillSelect($("index_file"), indexes);
  fillSelect($("audio"), audios);
});

$("unload").addEventListener("click", () => {
  $("model_file").value = "";
  $("index_file").value = "";
});

// ---------- Upload audio ----------
$("upload_audio").addEventListener("change", async e => {
  const f = e.target.files[0];
  if (!f) return;
  setStatus("uploading…", "busy");
  const r = await client.predict("/upload_audio", { file: handle_file(f) });
  const { audio, output } = r.data[0];
  $("audio").value = audio;
  $("output_path").value = output;
  setStatus("uploaded", "ok");
});

// ---------- Clear outputs ----------
$("clear_outputs").addEventListener("click", async () => {
  await client.predict("/delete_outputs", {});
  alert("Outputs cleared");
});

// ---------- Conditional UI ----------
$("infer_backing_vocals").addEventListener("change", e => {
  $("backing_vocals_box").classList.toggle("hidden", !e.target.checked);
});
$("deecho").addEventListener("change", e => {
  $("deeecho_model").classList.toggle("hidden", !e.target.checked);
});
$("denoise").addEventListener("change", e => {
  $("denoise_model").classList.toggle("hidden", !e.target.checked);
});
$("reverb").addEventListener("change", e => {
  document.querySelectorAll(".reverb-slider").forEach(el =>
    el.classList.toggle("hidden", !e.target.checked)
  );
});

// ---------- Full inference ----------
$("convert").addEventListener("click", async () => {
  setStatus("running…", "busy");
  $("vc_output1").value = "";
  try {
    const args = {
      model_file: $("model_file").value,
      index_file: $("index_file").value,
      audio: $("audio").value,
      output_path: $("output_path").value,
      export_format_rvc: $("export_format_rvc").value,
      split_audio: $("split_audio").checked,
      autotune: $("autotune").checked,
      vocal_model: $("vocal_model").value,
      karaoke_model: $("karaoke_model").value,
      dereverb_model: $("dereverb_model").value,
      deecho: $("deecho").checked,
      deeecho_model: $("deeecho_model").value,
      denoise: $("denoise").checked,
      denoise_model: $("denoise_model").value,
      reverb: $("reverb").checked,
      vocals_volume: +$("vocals_volume").value,
      instrumentals_volume: +$("instrumentals_volume").value,
      backing_vocals_volume: +$("backing_vocals_volume").value,
      export_format_final: $("export_format_final").value,
      devices: $("devices").value,
      pitch: +$("pitch").value,
      filter_radius: +$("filter_radius").value,
      index_rate: +$("index_rate").value,
      rms_mix_rate: +$("rms_mix_rate").value,
      protect: +$("protect").value,
      pitch_extract: $("pitch_extract").value,
      hop_length: +$("hop_length").value,
      reverb_room_size: +$("reverb_room_size").value,
      reverb_damping: +$("reverb_damping").value,
      reverb_wet_gain: +$("reverb_wet_gain").value,
      reverb_dry_gain: +$("reverb_dry_gain").value,
      reverb_width: +$("reverb_width").value,
      embedder_model: $("embedder_model").value,
      delete_audios: $("delete_audios").checked,
      use_tta: $("use_tta").checked,
      batch_size: +$("batch_size").value,
      infer_backing_vocals: $("infer_backing_vocals").checked,
      infer_backing_vocals_model: $("infer_backing_vocals_model").value,
      infer_backing_vocals_index: $("infer_backing_vocals_index").value,
      change_inst_pitch: +$("change_inst_pitch").value,
      pitch_back: +$("pitch_back").value,
      filter_radius_back: +$("filter_radius_back").value,
      index_rate_back: +$("index_rate_back").value,
      rms_mix_rate_back: +$("rms_mix_rate_back").value,
      protect_back: +$("protect_back").value,
      pitch_extract_back: $("pitch_extract_back").value,
      hop_length_back: +$("hop_length_back").value,
      export_format_rvc_back: $("export_format_rvc_back").value,
      split_audio_back: $("split_audio_back").checked,
      autotune_back: $("autotune_back").checked,
      embedder_model_back: $("embedder_model_back").value,
    };

    const r = await client.predict("/full_inference", args);
    const [info, audioPath] = r.data;
    $("vc_output1").value = info ?? "";
    if (audioPath) {
      // gradio returns a path or a FileData; normalize to URL
      const url = typeof audioPath === "string"
        ? "/audio_files/" + audioPath.replace(/^.*audio_files\//, "")
        : audioPath.url;
      $("vc_output2").src = url;
    }
    setStatus("done", "ok");
  } catch (err) {
    console.error(err);
    $("vc_output1").value = String(err);
    setStatus("error", "err");
  }
});

// ---------- Dl model ----------
$("download_model_btn").addEventListener("click", async () => {
  const link = $("model_url").value;
  if (!link) return;
  setStatus("downloading model…", "busy");
  const r = await client.predict("/download_model_url", { link });
  $("dlmodel_output").value = r.data[0];
  setStatus("done", "ok");
  const meta = await client.predict("/refresh_lists", {});
  fillSelect($("model_file"), meta.data[0].models);
  fillSelect($("index_file"), meta.data[0].indexes);
});

$("drop_model").addEventListener("change", async e => {
  const f = e.target.files[0];
  if (!f) return;
  setStatus("saving model…", "busy");
  const r = await client.predict("/save_drop_model", { dropbox: handle_file(f) });
  $("dlmodel_output").value = r.data[0];
  setStatus("saved", "ok");
  const meta = await client.predict("/refresh_lists", {});
  fillSelect($("model_file"), meta.data[0].models);
  fillSelect($("index_file"), meta.data[0].indexes);
});

// ---------- Dl music ----------
$("download_music_btn").addEventListener("click", async () => {
  const link = $("music_url").value;
  if (!link) return;
  setStatus("downloading music…", "busy");
  const r = await client.predict("/download_music_url", { link });
  $("dlmusic_output").value = r.data[0];
  setStatus("done", "ok");
  const meta = await client.predict("/refresh_lists", {});
  fillSelect($("audio"), meta.data[0].audios);
});

// ---------- Init ----------
loadMeta().then(() => setStatus("ready", "ok"));
