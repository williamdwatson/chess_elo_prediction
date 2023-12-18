import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Loader2 } from "lucide-react";
import { Textarea } from "@/components/ui/textarea";
import { ErrorAlert } from "./error_alert";
import createEmbedding from "./create_embedding";
import Instructions from "./instructions";
import indices from "./indices.json";
import * as ort from 'onnxruntime-web';
import Details from "./details";

/**
 * Move entry field and button
 * 
 * @component
 */
export default function MoveEntry() {
    const [badMoves, setBadMoves] = useState<{title: string, message: string}|null>(null);
    const [processing, setProcessing] = useState(false);
    const [whiteElo, setWhiteElo] = useState<number|null>(null);
    const [blackElo, setBlackElo] = useState<number|null>(null);
    const [_usedMoves, setUsedMoves] = useState<string[]|null>(null);
    const [usedTimeBase, setUsedTimeBase] = useState<number|null>(null);
    const [_usedTimeBonus, setUsedtimeBonus] = useState<number|null>(null);
    const inferenceWorkerRef = useRef<null|Worker>(null);

    /**
     * Comment regex
     */
    const COMMENT_PATTERN = /{.*?}/g;

    /**
     * Possible results to remove from the end of a game string
     */
    const possible_to_remove = [
        '1-0', '1- 0', '1 - 0', '1 -0', '0-1', '0- 1', '0 - 1', '0 -1',
        '0-0', '0- 0', '0 - 0', '0 -0', '1-1', '1- 1', '1 - 1', '1 -1',
        '½-½', '½- ½', '½ - ½', '½ -½', '0-½', '0- ½', '0 - ½', '0 -½',
        '½-0', '½- 0', '½ - 0', '½ -0', '1-½', '1- ½', '1 - ½', '1 -½',
        '½-1', '½- 1', '½ - 1', '½ -1'
    ];
    [...possible_to_remove].forEach(p => possible_to_remove.push(p.replace('-', '–')));
    [...possible_to_remove].filter(p => p.includes("½")).forEach(p => possible_to_remove.push(p.replace(/½/g, "1/2")));

    /**
     * Checks if a given string contains only characters allowed in a move
     * @param mv Move string to check
     * @returns Whether `mv` contains any banned characters or has fewer than two characters
     */
    const checkMoveOnly = (mv: string) => {
        return mv.length > 1 && ![".", "[", "]", "{", "}", "%", ":"].some(bad => mv.includes(bad));
    }

    /**
     * Trims down a move string by removing "!" (good/great move), "?" (bad move/blunder), "+" (check), "#" (checkmate), and ","
     * @param mv Move string
     * @returns `mv` with unnecessary characters removed
     */
    const pruneToMove = (mv: string) => {
        return mv.replace("!", "").replace("?", "").replace("+", "").replace(",", "").replace("#", "").trim();
    }

    /**
     * Checks if the input moves are valid, then runs the model
     */
    const runEstimation = () => {
        const el = document.getElementById("chess-moves") as HTMLTextAreaElement;
        const base_el = document.getElementById("time-base") as HTMLInputElement;
        const bonus_el = document.getElementById("time-bonus") as HTMLInputElement;
        if (el && base_el && bonus_el) {
            let val = el.value.replace(COMMENT_PATTERN, "").replace(/\./g, ". ").trim();
            const base_time = Number(base_el.value);
            const bonus_time = Number(bonus_el.value);
            if (val.length > 0) {
                for (const possible_result of possible_to_remove) {
                    if (val.endsWith(possible_result)) {
                        val = val.slice(0, val.length - possible_result.length).trimEnd();
                        break;
                    }
                }
                const current_moves = val.split(" ").filter(checkMoveOnly).map(pruneToMove);
                if (current_moves.length < 8) {
                    setBadMoves({title: "Too short", message: "At least 8 moves (4 per player) must be present for Elo estimation"});
                }
                else if (current_moves.length > 150) {
                    setBadMoves({title: "Too long", message: "No more than 150 moves (75 per player) can be present Elo estimation"});
                }
                else if (isNaN(base_time) || isNaN(bonus_time) || base_time < 0 || bonus_time < 0) {
                    setBadMoves({title: "Invalid time(s)", message: "Both the base and bonus times must be empty or positive numbers"});
                }
                else if (bonus_time !== 0 && base_time === 0) {
                    setBadMoves({title: "No base time", message: "If bonus time is set, then base time must also be set"});
                }
                else {
                    const embeddings = createEmbedding(current_moves, indices);
                    if (embeddings == null) {
                        setBadMoves({title: "Invalid moves", message: "Some of the moves present are invalid"});
                    }
                    else {
                        if (inferenceWorkerRef.current) {
                            setBadMoves(null);
                            setWhiteElo(null);
                            setBlackElo(null);
                            setProcessing(true);
                            setUsedMoves(current_moves);
                            setUsedTimeBase(base_time);
                            setUsedtimeBonus(bonus_time);
                            if (base_time !== 0) {
                                const base_time_tensor = new ort.Tensor("float32", [base_time], [1]);
                                const bonus_time_tensor = new ort.Tensor("float32", [bonus_time], [1]);
                                inferenceWorkerRef.current.postMessage([{arg0: embeddings[0], arg1: embeddings[1], arg2: base_time_tensor, arg3: bonus_time_tensor}, "avg_time"]);
                            }
                            else {
                                inferenceWorkerRef.current.postMessage([{arg0: embeddings[0], arg1: embeddings[1]}, "avg"]);
                            }
                        }
                    }
                }
            }
        }
    }

    // Upon load
    useEffect(() => {
        // Remove the annoying classes that mess with the layout
        document.getElementById("time-base")?.classList.remove("flex", "w-full");
        document.getElementById("time-bonus")?.classList.remove("flex", "w-full");
        
        // Set up the WebWorker for the inference
        // See https://github.com/jackylu0124/onnxruntime-web-worker-initialization-issue/tree/main/web-worker-001
        if (window.Worker) {
            inferenceWorkerRef.current = new Worker(new URL("inference.ts", import.meta.url), {type: "module"});
            inferenceWorkerRef.current.addEventListener("message", e => {
                setProcessing(false);
                const [b, w] = e.data.result.data.map((v: number) => Math.round(v));
                setWhiteElo(w);
                setBlackElo(b);
            });
        }
        
        Promise.all([fetch("model_avg.onnx"), fetch("model_avg_time.onnx")]).then(([onnx_file1, onnx_file2]) => {
            return Promise.all([onnx_file1.arrayBuffer(), onnx_file2.arrayBuffer()]);
        })
        .then(([onnx_file_buffer1, onnx_file_buffer2]) => {
            if (inferenceWorkerRef.current) {
                inferenceWorkerRef.current.postMessage([onnx_file_buffer1, onnx_file_buffer2, "initialize"], [onnx_file_buffer1, onnx_file_buffer2]);
            }
            else {
                console.error("Inference worker does not exist");
                setBadMoves({title: "Inference worker does not exist", message: "The WebWorker used for inference does not exist, and so the model cannot be loaded."});
            }
        })
        .catch((err) => {
            console.error(err);
            setBadMoves({title: "An error occurred", message: "An error occurred: " + err});
        });
    
        return () => {
            // Cleanup function
            if (inferenceWorkerRef.current) {
                inferenceWorkerRef.current.terminate();
            }
        };
    }, []);

    return (
        <div className="grid gap-2">
            {badMoves == null ? null : <ErrorAlert title={badMoves.title} message={badMoves.message}/>}
            <Instructions/>
            <Textarea placeholder="1. d4 d5 2. c4..." id="chess-moves" disabled={processing}/>
            <p>
                <span className="together"><Label htmlFor="time-base" style={{marginLeft: "25px", marginRight: "5px"}}>Base time (seconds):</Label>
                <Input type="number" id="time-base" min={0} style={{maxWidth: "10vw"}} placeholder="0" disabled={processing}/></span>
                <span className="together"><Label htmlFor="time-bonus" style={{marginLeft: "20px", marginRight: "5px"}}>Bonus time (seconds):</Label>
                <Input type="number" id="time-bonus" min={0} style={{maxWidth: "10vw"}} placeholder="0" disabled={processing}/></span>
            </p>
            <Button onClick={runEstimation} disabled={processing}>
                {processing ? <><Loader2 className="mr-2 h-4 w-4 animate-spin"/> Processing (this may take a minute)</>: <>Estimate Elos</>}
            </Button>
            {whiteElo == null ? null : <div><h2 style={{fontWeight: "bold", fontSize: "150%", display: "inline-block"}}>Estimated white Elo: {whiteElo}</h2> <Details which="white" pred={whiteElo} time={usedTimeBase == null}/></div>}
            {blackElo == null ? null : <div><h2 style={{fontWeight: "bold", fontSize: "150%", display: "inline-block"}}>Estimated black Elo: {blackElo}</h2> <Details which="black" pred={blackElo} time={usedTimeBase == null}/></div>}
        </div>
    )
}