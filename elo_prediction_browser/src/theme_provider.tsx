/**
 * See https://ui.shadcn.com/docs/dark-mode/vite
 */
import { createContext, useContext, useEffect, useState } from "react"

/**
 * Theme enumeration
 */
type Theme = "dark" | "light" | "system";

type ThemeProviderProps = {
    /**
     * Child components
     */
    children: React.ReactNode
    /**
     * Default theme
     */
    defaultTheme?: Theme
    /**
     * localStorage key to use for the theme
     */
    storageKey?: string
}
 
type ThemeProviderState = {
    /**
     * The current theme
     */
    theme: Theme
    /**
     * Sets the theme
     * @param theme New theme to use
     */
    setTheme: (theme: Theme) => void
}

/**
 * Default initial state ("system" theme)
 */
const initialState: ThemeProviderState = {
    theme: "system",
    setTheme: () => null
}
 
const ThemeProviderContext = createContext<ThemeProviderState>(initialState);

/**
 * The app theme provider
 * @param param0 Parameter containing:
 * * `children` - Subcomponents of the app
 * * `defaultTheme` - Default theme to use (default: "system")
 * * `storageKey` - Key to use in localStorage (default: "vite-ui-theme")
 */
export function ThemeProvider({
    children,
    defaultTheme = "system",
    storageKey = "vite-ui-theme",
    ...props
}: ThemeProviderProps) {
    const [theme, setTheme] = useState<Theme>(
        () => (localStorage.getItem(storageKey) as Theme) || defaultTheme
    )
 
    useEffect(() => {
        const root = window.document.documentElement
    
        root.classList.remove("light", "dark")
    
        if (theme === "system") {
            const systemTheme = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
        
            root.classList.add(systemTheme);
            return;
        }
    
        root.classList.add(theme);
    }, [theme]);
    
    const value = {
        theme,
        setTheme: (theme: Theme) => {
            localStorage.setItem(storageKey, theme);
            setTheme(theme);
        },
    }
    
    return (
        <ThemeProviderContext.Provider {...props} value={value}>
            {children}
        </ThemeProviderContext.Provider>
    )
}

/**
 * Set theme hook
 * @returns ThemeProviderContext state
 */
export const useTheme = () => {
    const context = useContext(ThemeProviderContext);
    
    if (context === undefined) {
        throw new Error("useTheme must be used within a ThemeProvider");
    }
    
    return context;
}