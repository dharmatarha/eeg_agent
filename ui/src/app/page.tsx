import { DataIntakeForm } from "@/components/data-intake-form";
import { RunHistory } from "@/components/run-history";

export default function HomePage() {
  return (
    <div className="min-h-screen flex">
      {/* Sidebar */}
      <aside className="w-64 border-r border-white/5 bg-slate-900/50 flex flex-col">
        <div className="p-4 border-b border-white/5">
          <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
            History
          </h2>
        </div>
        <div className="flex-1 overflow-y-auto">
          <RunHistory />
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 flex items-start justify-center p-8">
        <div className="w-full max-w-2xl space-y-8">
          {/* Header */}
          <div className="text-center space-y-3">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 text-indigo-400 text-xs font-medium border border-indigo-500/20 mb-2">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75" />
                <span className="relative inline-flex rounded-full h-2 w-2 bg-indigo-500" />
              </span>
              Multi-Agent System
            </div>
            <h1 className="text-4xl font-bold gradient-text">
              EEG Analysis Studio
            </h1>
            <p className="text-slate-400 text-sm max-w-md mx-auto leading-relaxed">
              AI-powered EEG pipeline with autonomous planning, execution,
              and quality assurance. Select your data and describe your analysis goals.
            </p>
          </div>

          {/* Form Card */}
          <div className="glass-card p-6">
            <DataIntakeForm />
          </div>

          {/* Footer info */}
          <div className="text-center text-xs text-slate-600 space-y-1">
            <p>
              Powered by MNE-Python • LangGraph • Gemini
            </p>
            <p>
              Data stays local — sandbox runs in Docker
            </p>
          </div>
        </div>
      </main>
    </div>
  );
}
