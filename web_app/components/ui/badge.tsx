import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "@/lib/utils"

const badgeVariants = cva(
    "inline-flex items-center rounded-full border px-2 py-0.5 text-[11px] font-bold tabular-nums transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
    {
        variants: {
            variant: {
                default: "border-transparent bg-primary text-primary-foreground",
                secondary: "border-border bg-muted text-muted-foreground",
                destructive: "border-down/25 bg-down/12 text-down",
                outline: "border-border text-foreground",
                // Semantic. Same geometry as every other pill; only the hue moves.
                success: "border-up/25 bg-up/12 text-up",
                warning: "border-warn/25 bg-warn/12 text-warn-ink",
                info: "border-primary/25 bg-primary/12 text-primary-ink",
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
