import { motion } from "framer-motion";
import { BarChart3, TrendingUp, Activity } from "lucide-react";

const GraphsTab = ({ visualizations }) => {
  if (!visualizations) {
    return (
      <div className="glass rounded-2xl p-12 text-center">
        <p className="text-xl text-gray-400">Графики не сгенерированы</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Orbital Diagram */}
      {visualizations.orbital_diagram && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="glass rounded-2xl p-6"
        >
          <div className="flex items-center gap-3 mb-4">
            <Activity className="w-6 h-6 text-cyan-400" />
            <h2 className="text-2xl font-bold text-white">Орбитальная диаграмма</h2>
          </div>
          <div className="flex justify-center bg-slate-900/50 rounded-xl p-4">
            <img
              src={visualizations.orbital_diagram}
              alt="Orbital Diagram"
              className="max-w-full h-auto rounded-lg"
              style={{ maxHeight: '600px' }}
            />
          </div>
          <p className="text-gray-400 text-center mt-4">
            Диаграмма показывает орбиту планеты вокруг звезды и зону обитаемости (зеленые пунктирные линии)
          </p>
        </motion.div>
      )}

      {/* Mass-Radius Chart */}
      {visualizations.mass_radius_chart && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="glass rounded-2xl p-6"
        >
          <div className="flex items-center gap-3 mb-4">
            <BarChart3 className="w-6 h-6 text-purple-400" />
            <h2 className="text-2xl font-bold text-white">Сравнение массы и радиуса</h2>
          </div>
          <div className="flex justify-center bg-slate-900/50 rounded-xl p-4">
            <img
              src={visualizations.mass_radius_chart}
              alt="Mass-Radius Comparison"
              className="max-w-full h-auto rounded-lg"
              style={{ maxHeight: '600px' }}
            />
          </div>
          <p className="text-gray-400 text-center mt-4">
            Сравнение массы и радиуса экзопланеты с планетами Солнечной системы
          </p>
        </motion.div>
      )}

      {/* Additional info */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.4 }}
        className="glass rounded-xl p-6"
      >
        <div className="flex items-center gap-3 mb-3">
          <TrendingUp className="w-6 h-6 text-cyan-400" />
          <h3 className="text-xl font-semibold text-white">О графиках</h3>
        </div>
        <div className="space-y-3 text-gray-300 text-sm">
          <p>
            <span className="text-cyan-300 font-semibold">Орбитальная диаграмма:</span> Показывает форму орбиты, расстояние до звезды
            и положение зоны обитаемости. Зона обитаемости - это область, где вода может существовать
            в жидком состоянии.
          </p>
          <p>
            <span className="text-cyan-300 font-semibold">Диаграмма масса-радиус:</span> Помогает определить тип планеты путем сравнения
            с известными планетами Солнечной системы. Логарифмическая шкала позволяет сравнивать
            планеты очень разных размеров.
          </p>
          <p>
            Все графики генерируются на основе реальных научных данных из астрономических баз данных.
          </p>
        </div>
      </motion.div>
    </div>
  );
};

export default GraphsTab;
