import { motion } from "framer-motion";
import { Brain, Sparkles, AlertCircle, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";

const AIAnalysisTab = ({ analysis, isAnalyzing, onAnalyze }) => {
  if (isAnalyzing) {
    return (
      <div className="glass rounded-2xl p-12 text-center">
        <Loader2 className="w-16 h-16 mx-auto mb-4 text-cyan-400 animate-spin" />
        <p className="text-xl text-white mb-2">Анализ данных...</p>
        <p className="text-gray-400">Обработка может занять несколько секунд</p>
      </div>
    );
  }

  if (!analysis) {
    return (
      <div className="space-y-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass rounded-2xl p-12 text-center"
        >
          <Brain className="w-20 h-20 mx-auto mb-6 text-purple-400" />
          <h2 className="text-2xl font-bold text-white mb-4">AI Анализ экзопланеты</h2>
          <p className="text-gray-400 mb-6 max-w-2xl mx-auto">
            Используйте мощь искусственного интеллекта для глубокого анализа данных о планете.
            AI проанализирует физические параметры, орбитальные характеристики и потенциальную обитаемость.
          </p>
          <Button
            onClick={onAnalyze}
            className="bg-gradient-to-r from-purple-500 to-pink-600 hover:from-purple-400 hover:to-pink-500 text-white px-8 py-6 text-lg"
          >
            <Sparkles className="w-5 h-5 mr-2" />
            Запустить AI анализ
          </Button>
        </motion.div>

        {/* Info about Ollama */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
          className="glass rounded-xl p-6 border border-yellow-500/20"
        >
          <div className="flex items-start gap-3">
            <AlertCircle className="w-6 h-6 text-yellow-400 flex-shrink-0 mt-1" />
            <div>
              <h3 className="text-lg font-semibold text-white mb-2">Требуется Ollama</h3>
              <p className="text-gray-400 text-sm mb-3">
                Для AI анализа необходимо, чтобы Ollama была запущена на вашем компьютере (http://localhost:11434).
                Если Ollama недоступна, анализ будет пропущен.
              </p>
              <a
                href="https://ollama.ai"
                target="_blank"
                rel="noopener noreferrer"
                className="text-cyan-400 hover:text-cyan-300 text-sm underline"
              >
                Узнать больше о Ollama →
              </a>
            </div>
          </div>
        </motion.div>
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="space-y-6"
    >
      <div className="glass rounded-2xl p-8">
        <div className="flex items-center gap-3 mb-6">
          <Brain className="w-8 h-8 text-purple-400" />
          <h2 className="text-3xl font-bold text-white">AI Анализ</h2>
        </div>
        <div className="prose prose-invert max-w-none">
          <div className="text-gray-300 whitespace-pre-wrap leading-relaxed">
            {analysis}
          </div>
        </div>
      </div>

      <div className="flex justify-center">
        <Button
          onClick={onAnalyze}
          variant="outline"
          className="border-cyan-500/30 text-cyan-400 hover:bg-cyan-500/10"
        >
          <Sparkles className="w-4 h-4 mr-2" />
          Перезапустить анализ
        </Button>
      </div>
    </motion.div>
  );
};

export default AIAnalysisTab;
