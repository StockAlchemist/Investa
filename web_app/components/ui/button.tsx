import * as React from "react"
import { Slot } from "@radix-ui/react-slot"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "@/lib/utils"

// Three heights, by density: h-7 (28) toolbars, h-9 (36) forms and nav,
// h-11 (44) touch and primary actions. Radius is the control step (8).
// Hover changes colour only — cards lift, controls do not.
const buttonVariants = cva(
    "inline-flex items-center justify-center gap-1.5 whitespace-nowrap rounded-control font-medium transition-colors disabled:pointer-events-none disabled:opacity-50 [&_svg]:size-4 [&_svg]:shrink-0",
    {
        variants: {
            variant: {
                default: "bg-primary text-primary-foreground font-semibold hover:bg-primary-hover",
                destructive: "bg-down-tint text-down font-semibold hover:bg-down/15",
                outline: "border border-border bg-card text-foreground hover:bg-muted",
                secondary: "border border-border bg-card text-foreground hover:bg-muted",
                ghost: "text-ink-2 hover:bg-muted hover:text-foreground",
                link: "text-primary underline-offset-4 hover:underline",
                glass: "bg-primary text-primary-foreground font-semibold hover:bg-primary-hover",
            },
            size: {
                default: "h-9 px-3.5 text-sm",
                sm: "h-7 px-2.5 text-xs",
                lg: "h-11 px-5 text-base",
                icon: "h-9 w-9",
                "icon-sm": "h-7 w-7",
            },
        },
        defaultVariants: {
            variant: "default",
            size: "default",
        },
    }
)

export interface ButtonProps
    extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
    asChild?: boolean
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
    ({ className, variant, size, asChild = false, ...props }, ref) => {
        const Comp = asChild ? Slot : "button"
        return (
            <Comp
                className={cn(buttonVariants({ variant, size, className }))}
                ref={ref}
                {...props}
            />
        )
    }
)
Button.displayName = "Button"

export { Button, buttonVariants }
