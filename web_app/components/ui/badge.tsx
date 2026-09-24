import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "@/lib/utils"

const badgeVariants = cva(
    "inline-flex items-center rounded-md border border-transparent px-2 py-0.5 text-xs font-semibold tabular-nums whitespace-nowrap transition-colors",
    {
        variants: {
            variant: {
                default: "bg-primary text-primary-foreground",
                secondary: "bg-muted text-ink-2",
                destructive: "bg-down-tint text-down",
                outline: "border-border text-ink-2",
                // Semantic. Same geometry as every other badge; only the hue moves, and
                // gain/loss figures inside one still carry their sign.
                success: "bg-up-tint text-up",
                warning: "bg-warn-tint text-warn-ink",
                info: "bg-primary-tint text-primary-ink",
            },
        },
        defaultVariants: {
            variant: "default",
        },
    }
)

export interface BadgeProps
    extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> { }

function Badge({ className, variant, ...props }: BadgeProps) {
    return (
        <div className={cn(badgeVariants({ variant }), className)} {...props} />
    )
}

export { Badge, badgeVariants }
