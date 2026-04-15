'use client';

import React, { useState, useRef, useEffect } from 'react';
import { 
    MessageCircle, 
    X, 
    Send, 
    Sparkles, 
    Loader2, 
    Maximize2, 
    Minimize2,
    RefreshCw,
    Trash2
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { sendChatMessage, ChatMessage } from '@/lib/api';
import { cn } from '@/lib/utils';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

export default function AIChat() {
    const [isOpen, setIsOpen] = useState(false);
    const [isMaximized, setIsMaximized] = useState(false);
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    
    // Initial welcome message (not stored in persistence)
    const WELCOME_MESSAGE: ChatMessage = { 
        role: 'ai', 
        text: "Hello! I'm Investa AI. How can I help you with your portfolio today?" 
    };
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const scrollRef = useRef<HTMLDivElement>(null);

    // Load history from localStorage on mount
    useEffect(() => {
        const saved = localStorage.getItem('investa_chat_history');
        if (saved) {
            try {
                setMessages(JSON.parse(saved));
            } catch (e) {
                console.error("Failed to load chat history", e);
            }
        }
    }, []);

    // Save history to localStorage on change
    useEffect(() => {
        if (messages.length > 0) {
            try {
                // Cap persistent history to last 50 messages to avoid QuotaExceededError
                const toSave = messages.slice(-50);
                localStorage.setItem('investa_chat_history', JSON.stringify(toSave));
            } catch (e) {
                console.warn("Storage quota exceeded, could not save full chat history", e);
                // If it still fails, we could try saving even fewer, 
                // but 50 small text messages should easily fit in 5MB.
            }
        }
    }, [messages]);

    // Auto-scroll to bottom of messages
    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [messages, isLoading]);

    const handleSend = async () => {
        if (!input.trim() || isLoading) return;

        const userMsg: ChatMessage = { role: 'user', text: input };
        setMessages(prev => [...prev, userMsg]);
        setInput('');
        setIsLoading(true);

        try {
            // Keep last 40 messages for context, filtering out the static welcome message
            // and ensuring we don't send an empty history if it's just text
            const history = messages
                .filter(m => m.text !== WELCOME_MESSAGE.text && m.text.trim() !== '')
                .slice(-40);
                
            const responseText = await sendChatMessage(input, history);
            
            // Safety: Don't add empty messages
            const finalReply = responseText?.trim() || "I encountered an issue generating a response. Please try again.";
            setMessages(prev => [...prev, { role: 'ai', text: finalReply }]);
        } catch (error) {
            console.error('Chat Error:', error);
            setMessages(prev => [...prev, { role: 'ai', text: "I'm sorry, I'm having trouble connecting to my brain right now. Please try again later." }]);
        } finally {
            setIsLoading(false);
        }
    };

    const clearHistory = () => {
        localStorage.removeItem('investa_chat_history');
        setMessages([]);
    };

    if (!isOpen) {
        return (
            <button
                onClick={() => setIsOpen(true)}
                className="fixed bottom-6 right-6 w-14 h-14 rounded-full bg-gradient-to-tr from-indigo-600 to-purple-600 text-white shadow-lg shadow-indigo-500/30 flex items-center justify-center hover:scale-110 active:scale-95 transition-all z-50 group overflow-hidden"
                aria-label="Open Investa AI Chat"
            >
                <div className="absolute inset-0 bg-white/10 opacity-0 group-hover:opacity-100 transition-opacity" />
                <Sparkles className="w-6 h-6 animate-pulse" />
            </button>
        );
    }

    return (
        <div 
            className={cn(
                "fixed z-50 transition-all duration-300 ease-in-out flex flex-col overflow-hidden",
                "bg-slate-50/90 dark:bg-slate-900/90 backdrop-blur-xl border border-white/20 dark:border-white/10 shadow-2xl rounded-3xl",
                isMaximized 
                    ? "inset-4 md:inset-10" 
                    : "bottom-6 right-6 w-[90vw] h-[70vh] md:w-[400px] md:h-[600px]"
            )}
        >
            {/* Header */}
            <div className="p-4 flex items-center justify-between border-b border-white/10 bg-white/5">
                <div className="flex items-center gap-3">
                    <div className="p-2 rounded-xl bg-gradient-to-tr from-indigo-500 to-purple-500 text-white">
                        <Sparkles className="w-4 h-4" />
                    </div>
                    <div>
                        <h3 className="text-sm font-bold bg-gradient-to-r from-indigo-500 to-purple-500 bg-clip-text text-transparent">Investa AI</h3>
                        <div className="flex items-center gap-1.5">
                            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                            <span className="text-[10px] text-muted-foreground font-medium uppercase tracking-wider">Online</span>
                            {messages.length > 0 && (
                                <>
                                    <span className="w-1 h-1 rounded-full bg-slate-400 mx-0.5" />
                                    <span className="text-[10px] text-indigo-500 font-bold uppercase tracking-wider">Memory Active</span>
                                </>
                            )}
                        </div>
                    </div>
                </div>
                <div className="flex items-center gap-1">
                    <Button variant="ghost" size="icon" className="h-8 w-8 rounded-lg" onClick={clearHistory} title="Clear conversation">
                        <Trash2 className="w-4 h-4 text-muted-foreground" />
                    </Button>
                    <Button variant="ghost" size="icon" className="h-8 w-8 rounded-lg hidden md:flex" onClick={() => setIsMaximized(!isMaximized)}>
                        {isMaximized ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
                    </Button>
                    <Button variant="ghost" size="icon" className="h-8 w-8 rounded-lg" onClick={() => setIsOpen(false)}>
                        <X className="w-4 h-4" />
                    </Button>
                </div>
            </div>

            {/* Message Area */}
            <div 
                ref={scrollRef}
                className="flex-1 p-4 space-y-4 overflow-y-auto scrollbar-thin scrollbar-thumb-white/10"
            >
                {messages.length === 0 && (
                    <div className="flex flex-col items-start max-w-[85%] animate-in fade-in slide-in-from-bottom-2 duration-300">
                        <div className="bg-white/40 dark:bg-white/5 border border-white/10 rounded-2xl rounded-tl-none px-4 py-3 text-sm leading-relaxed">
                            <div className="markdown-content">
                                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                    {WELCOME_MESSAGE.text}
                                </ReactMarkdown>
                            </div>
                        </div>
                        <span className="text-[10px] text-muted-foreground mt-1 px-1 font-medium uppercase tracking-tighter">
                            Investa AI
                        </span>
                    </div>
                )}
                {messages.map((msg, idx) => (
                    <div 
                        key={idx} 
                        className={cn(
                            "flex flex-col max-w-[85%] animate-in fade-in slide-in-from-bottom-2 duration-300",
                            msg.role === 'user' ? "ml-auto items-end" : "items-start"
                        )}
                    >
                        <div 
                            className={cn(
                                "px-4 py-3 rounded-2xl text-sm leading-relaxed",
                                msg.role === 'user' 
                                    ? "bg-indigo-600 text-white rounded-tr-none shadow-md shadow-indigo-600/20 selection:bg-white selection:text-indigo-600" 
                                    : "bg-white/40 dark:bg-white/5 border border-white/10 rounded-tl-none selection:bg-indigo-500/30"
                            )}
                        >
                            <div className={cn(
                                "markdown-content",
                                msg.role === 'user' ? "[&_*]:text-white" : ""
                            )}>
                                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                    {msg.text}
                                </ReactMarkdown>
                            </div>
                        </div>
                        <span className="text-[10px] text-muted-foreground mt-1 px-1 font-medium uppercase tracking-tighter">
                            {msg.role === 'user' ? 'You' : 'Investa AI'}
                        </span>
                    </div>
                ))}
                {isLoading && (
                    <div className="flex items-start gap-2 max-w-[85%] animate-in fade-in duration-300">
                        <div className="bg-white/40 dark:bg-white/5 border border-white/10 px-4 py-3 rounded-2xl rounded-tl-none">
                            <div className="flex gap-1 items-center h-4">
                                <span className="w-1.5 h-1.5 rounded-full bg-indigo-500/50 animate-bounce [animation-delay:-0.3s]" />
                                <span className="w-1.5 h-1.5 rounded-full bg-indigo-500/50 animate-bounce [animation-delay:-0.15s]" />
                                <span className="w-1.5 h-1.5 rounded-full bg-indigo-500/50 animate-bounce" />
                            </div>
                        </div>
                    </div>
                )}
            </div>

            {/* Input Area */}
            <div className="p-4 bg-white/5 border-t border-white/10">
                <form 
                    onSubmit={(e) => { e.preventDefault(); handleSend(); }}
                    className="flex gap-2"
                >
                    <Input 
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder="Ask about your portfolio..."
                        className="bg-white/5 border-white/10 rounded-xl focus-visible:ring-indigo-500 h-10 text-sm"
                        disabled={isLoading}
                    />
                    <Button 
                        type="submit" 
                        size="icon" 
                        className="bg-indigo-600 hover:bg-indigo-700 text-white h-10 w-10 shrink-0 rounded-xl transition-all"
                        disabled={isLoading || !input.trim()}
                    >
                        {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
                    </Button>
                </form>
                <p className="text-[9px] text-center text-muted-foreground/40 mt-3 uppercase tracking-[0.2em]">
                    Investa AI • Wealth Intelligence
                </p>
            </div>
        </div>
    );
}
