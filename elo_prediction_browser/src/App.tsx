import { ThemeProvider } from "./theme_provider";
import MoveEntry from "./move_entry";
import { ThemeToggle } from "./theme_toggle";
import "./App.css";

/**
 * The parent component
 * 
 * @component
 */
function App() {
    return (
        <ThemeProvider defaultTheme="dark" storageKey="vite-ui-theme">
            <h1 style={{fontWeight: "bolder", fontSize: "300%"}}>Elo Estimation</h1>
            <br/>
            <span style={{position: "fixed", top: "2%", right: "2%"}}><ThemeToggle/></span>
            <MoveEntry/>
        </ThemeProvider>
    )
}

export default App
