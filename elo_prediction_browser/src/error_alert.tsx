import { AlertCircle } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

interface ErrorAlertProps {
    /**
     * Title of the alert
     */
    title: string,
    /**
     * Main message of the alert
     */
    message: string
}

/**
 * Alert for showing an error
 * 
 * @component
 */
export function ErrorAlert(props: ErrorAlertProps) {
  return (
    <Alert variant="destructive">
        <AlertCircle className="h-4 w-4"/>
        <AlertTitle>{props.title}</AlertTitle>
        <AlertDescription>
            {props.message}
        </AlertDescription>
    </Alert>
  )
}