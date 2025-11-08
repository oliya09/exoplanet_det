import { motion } from "framer-motion";
import { Weight, Gauge, Thermometer, Circle } from "lucide-react";

const ParamCard = ({ icon: Icon, label, value, unit, color = "cyan" }) => {
  const colorMap = {
    cyan: "from-cyan-500/20 to-cyan-600/20 border-cyan-500/30",
    blue: "from-blue-500/20 to-blue-600/20 border-blue-500/30",
    purple: "from-purple-500/20 to-purple-600/20 border-purple-500/30",
    orange: "from-orange-500/20 to-orange-600/20 border-orange-500/30",
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className={`glass rounded-xl p-6 bg-gradient-to-br ${colorMap[color]} border hover:scale-105 transition-transform`}
    >
      <div className="flex items-center gap-3 mb-3">
        <Icon className="w-6 h-6 text-white" />
        <h3 className="text-lg font-semibold text-white">{label}</h3>
      </div>
      <p className="text-3xl font-bold text-white">
        {value !== null && value !== undefined ? value.toFixed(3) : "—"}
      </p>
      <p className="text-sm text-gray-300 mt-1">{unit}</p>
    </motion.div>
  );
};

const PhysicalTab = ({ data }) => {
  if (!data || Object.keys(data).length === 0) {
    return (
      <div className="glass rounded-2xl p-12 text-center">
        <p className="text-xl text-gray-400">Физические параметры не найдены</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Mass Section */}
      <div>
        <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
          <Weight className="w-6 h-6 text-cyan-400" />
          Масса
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <ParamCard
            icon={Weight}
            label="Масса (Земли)"
            value={data.mass || data.mass_eu}
            unit="M⊕ (массы Земли)"
            color="cyan"
          />
          <ParamCard
            icon={Weight}
            label="Масса (Юпитера)"
            value={data.mass_jupiter}
            unit="MJ (массы Юпитера)"
            color="blue"
          />
        </div>
      </div>

      {/* Radius Section */}
      <div>
        <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
          <Circle className="w-6 h-6 text-purple-400" />
          Радиус
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <ParamCard
            icon={Circle}
            label="Радиус (Земли)"
            value={data.radius || data.radius_eu}
            unit="R⊕ (радиусы Земли)"
            color="purple"
          />
          <ParamCard
            icon={Circle}
            label="Радиус (Юпитера)"
            value={data.radius_jupiter}
            unit="RJ (радиусы Юпитера)"
            color="blue"
          />
        </div>
      </div>

      {/* Other Physical Properties */}
      <div>
        <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
          <Gauge className="w-6 h-6 text-orange-400" />
          Другие характеристики
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          <ParamCard
            icon={Gauge}
            label="Плотность"
            value={data.density}
            unit="g/cm³"
            color="orange"
          />
          <ParamCard
            icon={Gauge}
            label="Гравитация"
            value={data.gravity}
            unit="m/s²"
            color="cyan"
          />
          <ParamCard
            icon={Thermometer}
            label="Равновесная температура"
            value={data.equilibrium_temp || data.temp_calculated}
            unit="K (Кельвин)"
            color="orange"
          />
        </div>
      </div>

      {/* Comparison with Solar System */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
        className="glass rounded-xl p-6"
      >
        <h3 className="text-xl font-semibold text-white mb-4">Сравнение с планетами Солнечной системы</h3>
        <div className="space-y-3 text-gray-300">
          {data.mass && (
            <p>
              {data.mass < 0.5 && "Легче Земли, сравнима с Меркурием или Марсом"}
              {data.mass >= 0.5 && data.mass < 5 && "Сравнима с Землей, супер-Земля"}
              {data.mass >= 5 && data.mass < 50 && "Сравнима с Нептуном или Ураном, ледяной гигант"}
              {data.mass >= 50 && data.mass < 500 && "Сравнима с Юпитером или Сатурном, газовый гигант"}
              {data.mass >= 500 && "Тяжелее Юпитера, возможно коричневый карлик"}
            </p>
          )}
          {data.radius && (
            <p>
              {data.radius < 1.5 && "Размер как у Земли, скалистая планета"}
              {data.radius >= 1.5 && data.radius < 4 && "Супер-Земля или мини-Нептун"}
              {data.radius >= 4 && data.radius < 8 && "Размер как у Нептуна или Урана"}
              {data.radius >= 8 && "Размер как у Юпитера или больше"}
            </p>
          )}
        </div>
      </motion.div>
    </div>
  );
};

export default PhysicalTab;
