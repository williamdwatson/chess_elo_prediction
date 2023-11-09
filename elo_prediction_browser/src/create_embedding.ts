import * as ort from 'onnxruntime-web';

/**
 * Mapping of move strings to embeddings
 */
const embeddings: Record<string, number[]> = {};

/**
 * Creates the Tensor embeddings of the given `move_list`
 * @param move_list List of string moves in algebraic notation
 * @returns Length-2 array of the embedding as a [150, 100] float32 Tensor and the mask as a 150-length boolean tensor;
 * or `null` if an invalid move is present in `move_list`
 */
export default function createEmbedding(move_list: string[]) {
    const moves = move_list.map(mv => embeddings[mv]);
    if (moves.some(mv => mv == undefined)) {
        return null;
    }
    while (moves.length < 150) {
        moves.push(Array(100).fill(0));
    }
    const arr = new ort.Tensor("float32", moves.flat(), [150, 100]);
    const mask = new ort.Tensor("bool", [...Array(150).keys()].map(i => i < move_list.length ? false : true), [150]);
    return [arr, mask];
}