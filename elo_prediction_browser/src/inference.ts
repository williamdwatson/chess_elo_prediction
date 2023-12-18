import * as ort from "onnxruntime-web";

var session_avg: ort.InferenceSession | undefined = undefined;
var session_avg_time: ort.InferenceSession | undefined = undefined;

onmessage = async (e) => {
    const what = e.data[e.data.length-1];
    if (what === "initialize") {
        const onnx_file_buffer1 = e.data[0];
        const onnx_file_buffer2 = e.data[1];
        try {
            if (session_avg == undefined) {
                session_avg = await ort.InferenceSession.create(onnx_file_buffer1, {executionProviders: ["wasm"]});
            }
            if (session_avg_time == undefined ){
                session_avg_time = await ort.InferenceSession.create(onnx_file_buffer2, {executionProviders: ["wasm"]});
            }
        }
        catch (err) {
            console.error(err);
        }
    }
    else {
        const inference_args = e.data[0];
        try {
            if (what === "avg" && session_avg != undefined) {
                const result = (await session_avg.run(inference_args)).others_1;
                postMessage({result: result, what: what});
            }
            else if (what === "avg_time" && session_avg_time != undefined) {
                const result = (await session_avg_time.run(inference_args)).others_1;
                postMessage({result: result, what: what});
            }
        }
        catch (err) {
            console.error(err);
        }
    }
}