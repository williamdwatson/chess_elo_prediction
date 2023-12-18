import * as ort from 'onnxruntime-web';


/**
 * Creates the Tensor embeddings of the given `move_list`
 * @param move_list List of string moves in algebraic notation
 * @param indices_mapping Mapping of move strings to indices
 * @returns Integer move indices as a 150-length int64 tensor and the mask as a 150-length boolean tensor;
 * or `null` if an invalid move is present in `move_list`
 */
export default function createEmbedding(move_list: string[], indices_mapping: Record<string, number>) {
    const moves = move_list.map(mv => indices_mapping[mv]);
    if (moves.some(mv => mv == undefined)) {
        return null;
    }
    while (moves.length < 150) {
        moves.push(0);
    }
    const arr = new ort.Tensor("int64", moves, [1, 150]);
    const mask = new ort.Tensor("bool", [...Array(150).keys()].map(i => i < move_list.length), [1, 150]);
    return [arr, mask];
}