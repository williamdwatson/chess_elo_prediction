import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import avg_percentiles_black from "./avg_percentiles_black.json";
import avg_percentiles_white from "./avg_percentiles_white.json";
import avg_time_percentiles_black from "./avg_time_percentiles_black.json";
import avg_time_percentiles_white from "./avg_time_percentiles_white.json";

interface DetailsProps {
    /**
     * Which color player the predicted Elo is for
     */
    which: "black"|"white",
    /**
     * Whether the game included time information or not
     */
    time: boolean,
    /**
     * The predicted Elo
     */
    pred: number
}
/**
 * Statistics about validation games with the predicted Elo
 */
interface stats_t {
    /**
     * The number of validation games that had the predicted Elo
     */
    number: number,
    /**
     * The minimum actual Elo of validation games that had the predicted Elo
     */
    min: number,
    /**
     * The maximum actual Elo of validation games that had the predicted Elo
     */
    max: number,
    /**
     * The average actual Elo of validation games that had the predicted Elo
     */
    mean: number,
    /**
     * The median actual Elo of validation games that had the predicted Elo
     */
    median: number,
    /**
     * The 25th percentile of actual Elos of validation games that had the predicted Elo
     */
    "25th": number,
    /**
     * The 75th percentile of actual Elos of validation games that had the predicted Elo
     */
    "75th": number,
    /**
     * The 10th percentile of actual Elos of validation games that had the predicted Elo
     */
    "10th": number,
    /**
     * The 90th percentile of actual Elos of validation games that had the predicted Elo
     */
    "90th": number,
    /**
     * The 5th percentile of actual Elos of validation games that had the predicted Elo
     */
    "5th": number,
    /**
     * The 95th percentile of actual Elos of validation games that had the predicted Elo
     */
    "95th": number,
    /**
     * The 0.5th percentile of actual Elos of validation games that had the predicted Elo
     */
    "0.5th": number,
    /**
     * The 99.5th percentile of actual Elos of validation games that had the predicted Elo
     */
    "99.5th": number
}

export default function Details(props: DetailsProps) {
    const vals: Array<[string, keyof stats_t]> = [
        ["Average", "mean"],
        ["Median", "median"],
        ["Minimum", "min"],
        ["Maximum", "max"]
    ];
    ["25th", "75th", "10th", "90th", "5th", "95th", "0.5th", "99.5th"].forEach(v => vals.push([v + " percentile", v as keyof stats_t]));
    
    /**
     * Gets the value for the key closest to `val`
     * @param val Key not present in `statistics`
     * @param statistics Mapping of numeric keys (or string versions thereof) to stats objects
     * @returns The value for the key in `statistics` that is closest numerically to `val`
     */
    const getClosest = (val: number, statistics: Record<string, stats_t>): [stats_t, string] => {
        let closest_key = Object.keys(statistics)[0];
        let closest_key_diff = Math.abs(Number(closest_key) - val);
        Object.keys(statistics).forEach(k => {
            const key_val = Number(k);
            if (Math.abs(key_val - val) < closest_key_diff) {
                closest_key = k;
                closest_key_diff = Math.abs(key_val - val);
            }
        });
        return [statistics[closest_key], closest_key];
    }

    let stats: stats_t;
    let closest_elo = "-1";
    if (props.which === "black" && !props.time) {
        stats = (avg_percentiles_black as Record<string, stats_t>)[props.pred.toString()];
        if (stats == undefined) {
            [stats, closest_elo] = getClosest(props.pred, avg_percentiles_black);
        }
    }
    else if (props.which === "white" && !props.time) {
        stats = (avg_percentiles_white as Record<string, stats_t>)[props.pred.toString()];
        if (stats == undefined) {
            [stats, closest_elo] = getClosest(props.pred, avg_percentiles_white);
        }
    }
    else if (props.which === "black" && props.time) {
        stats = (avg_time_percentiles_black as Record<string, stats_t>)[props.pred.toString()];
        if (stats == undefined) {
            [stats, closest_elo] = getClosest(props.pred, avg_time_percentiles_black);
        }
    }
    else {
        stats = (avg_time_percentiles_white as Record<string, stats_t>)[props.pred.toString()];
        if (stats == undefined) {
            [stats, closest_elo] = getClosest(props.pred, avg_time_percentiles_white);
        }
    }
    
    if (stats.number === 1) {
        return (
            <Dialog>
                <DialogTrigger asChild>
                    <Button variant="link">Details</Button>
                </DialogTrigger>
                <DialogContent className="sm:max-w-[425px]">
                    <DialogHeader>
                        <DialogTitle>Details for predicted Elo {props.pred}</DialogTitle>
                        <DialogDescription>
                            Statistics on the game in the validation set that had this predicted Elo for {props.which}
                        </DialogDescription>
                    </DialogHeader>
                    {closest_elo !== "-1" ? 
                    "There were no games in the validation set with a predicted Elo of " + props.pred + " for " + props.which + ". The closest predicted Elo was " + closest_elo + ", and the "
                    : "There was only one game in the validation set with this predicted Elo. The "}
                    actual Elo for that game was <b>{stats.median}</b>.
                </DialogContent>
            </Dialog>
        )
    }
    else {
        return (
            <Dialog>
                <DialogTrigger asChild>
                    <Button variant="link">Details</Button>
                </DialogTrigger>
                <DialogContent className="sm:max-w-[425px]">
                    <DialogHeader>
                        <DialogTitle>Details for predicted Elo {props.pred}</DialogTitle>
                        <DialogDescription>
                            Statistics on games in the validation set that had this predicted Elo for {props.which}
                        </DialogDescription>
                    </DialogHeader>
                    {closest_elo !== "-1" ?
                    "There were no games in the validation set with a predicted Elo of " + props.pred + " for " + props.which + ". The closest predicted Elo was " + closest_elo + "."
                    : null}
                    There were {stats.number.toLocaleString()} validation games with this predicted Elo {closest_elo === "-1" ? "for " + props.which : null}.
                    <table>
                        <tr>
                            <th>Statistic</th>
                            <th>Actual Elo</th>
                        </tr>
                        {vals.map(([label, attr]) => {
                            return (
                                <tr>
                                    <td style={{textAlign: "center"}}>{label}</td>
                                    <td style={{textAlign: "center"}}>{label === "Number" ? stats?.number.toLocaleString() : stats![attr]}</td>
                                </tr>
                            )
                        })}
                    </table>
                </DialogContent>
            </Dialog>
        )
    }
}
  