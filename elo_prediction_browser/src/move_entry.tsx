import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Loader2 } from "lucide-react";
import { Textarea } from "@/components/ui/textarea";
import { ErrorAlert } from "./error_alert";
import createEmbedding from "./create_embedding";
import Instructions from "./instructions";

/**
 * Move entry field and button
 * 
 * @component
 */
export default function MoveEntry() {
    const [badMoves, setBadMoves] = useState<{title: string, message: string}|null>(null);
    const [processing, setProcessing] = useState(false);

    /**
     * Comment regex
     */
    const COMMENT_PATTERN = /{.*?}/;

    /**
     * Checks if a given string contains only characters allowed in a move
     * @param mv Move string to check
     * @returns Whether `mv` contains any banned characters or has fewer than two characters
     */
    const checkMoveOnly = (mv: string) => {
        return mv.length > 1 && ![".", "[", "]", "{", "}", "%", ":"].some(bad => mv.includes(bad));
    }

    /**
     * Trims down a move string by removing "!" (good/great move), "?" (bad move/blunder), "+" (check) and ","
     * @param mv Move string
     * @returns `mv` with unnecessary characters removed
     */
    const pruneToMove = (mv: string) => {
        return mv.replace("!", "").replace("?", "").replace("+", "").replace(",", "").trim();
    }

    /**
     * Checks if the input moves are valid, then runs the model
     */
    const runEstimation = () => {
        const el = document.getElementById("chess-moves") as HTMLTextAreaElement;
        const base_el = document.getElementById("time-base") as HTMLInputElement;
        const bonus_el = document.getElementById("time-bonus") as HTMLInputElement;
        if (el && base_el && bonus_el) {
            const val = el.value.replace(COMMENT_PATTERN, "");
            const base_time = Number(base_el.value);
            const bonus_time = Number(bonus_el.value);
            if (val.length > 0) {
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
                    const embeddings = createEmbedding(current_moves);
                    if (embeddings == null) {
                        setBadMoves({title: "Invalid moves", message: "Some of the moves present are invalid"});
                    }
                    else {
                        setBadMoves(null);
                        setProcessing(true);
                        // TODO Add inference here, possibly using WebWorkers if it's too slow
                    }
                }
            }
        }
    }

    // Upon load, remove the annoying classes that mess with the layout
    useEffect(() => {
        document.getElementById("time-base")?.classList.remove("flex", "w-full");
        document.getElementById("time-bonus")?.classList.remove("flex", "w-full");
    }, []);

    return (
        <div className="grid gap-2">
            {badMoves == null ? null : <ErrorAlert title={badMoves.title} message={badMoves.message}/>}
            <Instructions/>
            <Textarea placeholder="1. d4 d5 2. c4..." id="chess-moves"/>
            <p>
                Model options:
                <Checkbox id="include-checkmate" style={{marginLeft: "15px", marginRight: "5px", position: "relative", top: "2px"}}/>
                <Label htmlFor="include-checkmate">Include checkmate</Label>
                <Label htmlFor="time-base" style={{marginLeft: "25px", marginRight: "5px"}}>Base time (seconds):</Label>
                <Input type="number" id="time-base" min={0} style={{maxWidth: "10%"}} placeholder="0"/>
                <Label htmlFor="time-bonus" style={{marginLeft: "20px", marginRight: "5px"}}>Bonus time (seconds):</Label>
                <Input type="number" id="time-bonus" min={0} style={{maxWidth: "10%"}} placeholder="0"/>
            </p>
            <Button onClick={runEstimation} disabled={processing}>
                {processing ? <><Loader2 className="mr-2 h-4 w-4 animate-spin"/> Processing</>: <>Estimate Elos</>}
            </Button>
        </div>
    )
}