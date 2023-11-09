import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";

export default function Instructions() {
    return (
        <Dialog>
            <DialogTrigger asChild>
                <Button variant="link">Instructions</Button>
            </DialogTrigger>
            <DialogContent className="sm:max-w-[425px]">
                <DialogHeader>
                    <DialogTitle>Instructions</DialogTitle>
                    <DialogDescription>
                        How to input a game for Elo prediction
                    </DialogDescription>
                </DialogHeader>
                <ul style={{listStyleType: "circle", padding: "0 15px"}}>
                    <li>The game should be input in algebraic notation, with each move separated by a space.</li>
                    <li>Move numbers (for instance the "1." in <i>1. e4 e5</i>) are permitted but not required.</li>
                    <li>Comments (enclosed in curly braces) will be ignored, as will move strength notation ("!", "?", and their derivatives).</li>
                    <li>"+"s (indicating checks) are likewise ignored.</li>
                    <li>Depending on the model selection, placing a "#" at the end (indicating checkmate) may or may not be ignored.</li>
                    <li>Only games between 8 and 150 moves (i.e. plies, inclusive; that is, between 4 and 75 moves per player) can be used.</li>
                    <li>If base time is 0, then time will not be used as a feature.</li>
                </ul>
            </DialogContent>
        </Dialog>
    )
}
  