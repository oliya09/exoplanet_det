import { motion } from "framer-motion";
import { Star, Thermometer, Zap, Ruler, Clock } from "lucide-react";

const StarParam = ({ icon: Icon, label, value, unit, color = "yellow" }) => {
  const colorMap = {
    yellow: "from-yellow-500/20 to-orange-500/20 border-yellow-500/30",
    blue: "from-blue-500/20 to-cyan-500/20 border-blue-500/30",
    red: "from-red-500/20 to-orange-500/20 border-red-500/30",
    purple: "from-purple-500/20 to-pink-500/20 border-purple-500/30",
  };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3 }}
      className={`glass rounded-xl p-6 bg-gradient-to-br ${colorMap[color]} border hover:scale-105 transition-transform`}
    >
      <div className="flex items-center gap-3 mb-3">
        <Icon className="w-6 h-6 text-white" />
        <h3 className="text-lg font-semibold text-white">{label}</h3>
      </div>
      <p className="text-2xl font-bold text-white">
        {value !== null && value !== undefined ? value : "—"}
      </p>
      {unit && <p className="text-sm text-gray-300 mt-1">{unit}</p>}
    </motion.div>
  );
};

const HostStarTab = ({ data }) => {
  if (!data || Object.keys(data).length === 0) {
    return (
      <div className="glass rounded-2xl p-12 text-center">
        <p className="text-xl text-gray-400">Данные о звезде-хозяине не найдены</p>
      </div>
    );
  }

  const mass = data.mass || data.star_mass_eu;
  const radius = data.radius || data.star_radius_eu;
  const temperature = data.temperature;
  const luminosity = data.luminosity;
  const metallicity = data.metallicity || data.star_metallicity_eu;
  const age = data.age;
  const distance = data.distance;

  return (
    <div className="space-y-6">
      {/* Star name and type */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass rounded-2xl p-8 glow"
      >
        <div className="flex items-center gap-3 mb-4">
          <Star className="w-8 h-8 text-yellow-400" />
          <h2 className="text-3xl font-bold text-white">{data.name || "Неизвестная звезда"}</h2>
        </div>
        {data.spectral_type && (
          <p className="text-lg text-gray-300">
            Спектральный класс: <span className="text-cyan-300 font-semibold">{data.spectral_type}</span>
          </p>
        )}
      </motion.div>

      {/* Physical parameters */}
      <div>
        <h3 className="text-2xl font-bold text-white mb-4">Физические характеристики</h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          <StarParam
            icon={Star}
            label="Масса"
            value={mass ? mass.toFixed(3) : null}
            unit="масс Солнца (M☉)"
            color="yellow"
          />
          <StarParam
            icon={Ruler}
            label="Радиус"
            value={radius ? radius.toFixed(3) : null}
            unit="радиусов Солнца (R☉)"
            color="blue"
          />
          <StarParam
            icon={Thermometer}
            label="Температура"
            value={temperature ? temperature.toFixed(0) : null}
            unit="K (Кельвин)"
            color="red"
          />
          <StarParam
            icon={Zap}
            label="Светимость"
            value={luminosity ? luminosity.toFixed(3) : null}
            unit="log(L/L☉)"
            color="yellow"
          />
          <StarParam
            icon={Star}
            label="Металличность"
            value={metallicity ? metallicity.toFixed(3) : null}
            unit="[Fe/H] dex"
            color="purple"
          />
          <StarParam
            icon={Clock}
            label="Возраст"
            value={age ? age.toFixed(2) : null}
            unit="млрд лет (Gyr)"
            color="blue"
          />
        </div>
      </div>

      {/* Distance */}
      {distance && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
          className="glass rounded-xl p-6"
        >
          <h3 className="text-xl font-semibold text-white mb-3">Расстояние от Земли</h3>
          <p className="text-3xl font-bold text-cyan-300 mb-2">{distance.toFixed(2)} парсек</p>
          <p className="text-gray-400">~ {(distance * 3.26).toFixed(2)} световых лет</p>
        </motion.div>
      )}

      {/* Detailed description */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
        className="glass rounded-xl p-6"
      >
        <h3 className="text-2xl font-bold text-white mb-4">Описание</h3>
        <div className="space-y-3 text-gray-300">
          {temperature && (
            <p>
              {temperature > 30000 && "Это очень горячая голубая звезда класса O."}
              {temperature <= 30000 && temperature > 10000 && "Это горячая голубовато-белая звезда класса B."}
              {temperature <= 10000 && temperature > 7500 && "Это белая звезда класса A."}
              {temperature <= 7500 && temperature > 6000 && "Это желтовато-белая звезда класса F."}
              {temperature <= 6000 && temperature > 5200 && "Это желтая звезда класса G, похожая на наше Солнце."}
              {temperature <= 5200 && temperature > 3700 && "Это оранжевая звезда класса K."}
              {temperature <= 3700 && "Это красная звезда класса M - холодный красный карлик."}
            </p>
          )}

          {mass && (
            <p>
              Масса звезды составляет {(mass * 100).toFixed(0)}% от массы Солнца.
              {mass < 0.5 && " Это небольшая звезда, красный карлик."}
              {mass >= 0.5 && mass < 0.8 && " Меньше Солнца, но достаточно стабильная."}
              {mass >= 0.8 && mass < 1.2 && " Примерно как Солнце!"}
              {mass >= 1.2 && " Более массивная чем Солнце."}
            </p>
          )}

          {metallicity !== null && metallicity !== undefined && (
            <p>
              Металличность [Fe/H] = {metallicity.toFixed(3)}
              {metallicity > 0 && " означает, что звезда богаче металлами, чем Солнце."}
              {metallicity < 0 && " означает, что звезда беднее металлами, чем Солнце - возможно, это старая звезда."}
              {Math.abs(metallicity) < 0.1 && " Близко к солнечному значению."}
            </p>
          )}

          {age && (
            <p>
              Возраст звезды около {age.toFixed(2)} миллиардов лет.
              {age < 1 && " Это молодая звезда."}
              {age >= 1 && age < 5 && " Примерно как Солнце (4.6 млрд лет)."}
              {age >= 5 && age < 10 && " Это старая звезда."}
              {age >= 10 && " Очень старая звезда, почти как Вселенная."}
            </p>
          )}
        </div>
      </motion.div>
    </div>
  );
};

export default HostStarTab;
