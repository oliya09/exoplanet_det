// src/components/tabs/OverviewTab.jsx
import { motion } from "framer-motion";
import { Info, Calendar, Target, MapPin, BarChart2, AlertCircle, CheckCircle, Globe } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";

const DataCard = ({ icon: Icon, label, value, color = "cyan" }) => {
  const colorClasses = {
    cyan: "text-cyan-400 border-cyan-500/30",
    purple: "text-purple-400 border-purple-500/30",
    blue: "text-blue-400 border-blue-500/30",
    green: "text-green-400 border-green-500/30",
  };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3 }}
      className={`glass rounded-xl p-6 border ${colorClasses[color]} hover:scale-105 transition-transform`}
    >
      <div className="flex items-start gap-4">
        <div className={`${colorClasses[color].split(' ')[0]} p-3 rounded-lg bg-opacity-10`}>
          <Icon className="w-6 h-6" />
        </div>
        <div className="flex-1">
          <p className="text-sm text-gray-400 mb-1">{label}</p>
          <p className="text-lg font-semibold text-white">{value || "Нет данных"}</p>
        </div>
      </div>
    </motion.div>
  );
};

const OverviewTab = ({ data }) => {
  const discoveryInfo = data.discovery_info || {};
  const physicalParams = data.physical_params || {};
  const orbitalParams = data.orbital_params || {};
  const hostStar = data.host_star || {};
  const habitability = data.habitability_score || {};
  const { total_score = 0, survival_chance = 0, category = "Unknown", factors = {}, recommendations = [], risks = [] } = habitability;

  const getCategoryColor = (cat) => {
    switch (cat.toLowerCase()) {
      case "earth-like": return "bg-green-500/20 text-green-400 border-green-500/30";
      case "promising": return "bg-yellow-500/20 text-yellow-400 border-yellow-500/30";
      case "moderate": return "bg-orange-500/20 text-orange-400 border-orange-500/30";
      case "challenging": return "bg-red-500/20 text-red-400 border-red-500/30";
      case "hostile": return "bg-purple-500/20 text-purple-400 border-purple-500/30";
      default: return "bg-gray-500/20 text-gray-400 border-gray-500/30";
    }
  };

  return (
    <div className="space-y-6">
      {/* === Habitability & Survival Score === */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="grid grid-cols-1 md:grid-cols-2 gap-6"
      >
        {/* Habitability Score */}
        <Card className="glass border border-cyan-500/20 p-6 glow-strong">
          <div className="flex items-center gap-3 mb-4">
            <BarChart2 className="w-6 h-6 text-cyan-400" />
            <h2 className="text-2xl font-bold text-white">Обитаемость</h2>
          </div>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Оценка:</span>
              <span className="text-2xl font-bold text-cyan-300">{total_score}/100</span>
            </div>
            <Progress 
              value={total_score} 
              className="h-4 bg-cyan-900/50"
              indicatorClassName="bg-gradient-to-r from-cyan-500 to-blue-500"
            />
            <Badge className={`${getCategoryColor(category)} px-3 py-1 text-sm font-medium`}>
              {category}
            </Badge>
          </div>
        </Card>

        {/* Survival Chance */}
        <Card className="glass border border-purple-500/20 p-6 glow-strong">
          <div className="flex items-center gap-3 mb-4">
            <Globe className="w-6 h-6 text-purple-400" />
            <h2 className="text-2xl font-bold text-white">Шанс выживания</h2>
          </div>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Вероятность:</span>
              <span className="text-2xl font-bold text-purple-300">{survival_chance}%</span>
            </div>
            <Progress 
              value={survival_chance} 
              className="h-4 bg-purple-900/50"
              indicatorClassName="bg-gradient-to-r from-purple-500 to-pink-500"
            />
            <p className="text-sm text-gray-400">На основе 5 ключевых факторов</p>
          </div>
        </Card>
      </motion.div>

      {/* === Main Info Banner === */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass rounded-2xl p-8 border border-cyan-500/20"
      >
        <div className="flex items-center gap-3 mb-6">
          <Info className="w-8 h-8 text-cyan-400" />
          <h2 className="text-3xl font-bold text-white">{data.name}</h2>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h3 className="text-lg font-semibold text-cyan-300 mb-3">Основная информация</h3>
            <div className="space-y-2 text-gray-300">
              <p><span className="text-gray-400">Звезда-хозяин:</span> <span className="text-white font-medium">{hostStar.name || "Неизвестно"}</span></p>
              <p><span className="text-gray-400">Метод обнаружения:</span> <span className="text-white font-medium">{discoveryInfo.discovery_method || "Нет данных"}</span></p>
              <p><span className="text-gray-400">Год открытия:</span> <span className="text-white font-medium">{discoveryInfo.discovery_year || "Нет данных"}</span></p>
              {discoveryInfo.discovery_facility && (
                <p><span className="text-gray-400">Обсерватория:</span> <span className="text-white font-medium">{discoveryInfo.discovery_facility}</span></p>
              )}
            </div>
          </div>
          
          <div>
            <h3 className="text-lg font-semibold text-cyan-300 mb-3">Краткие характеристики</h3>
            <div className="space-y-2 text-gray-300">
              <p><span className="text-gray-400">Масса:</span> <span className="text-white font-medium">{physicalParams.pl_bmasse ? `${parseFloat(physicalParams.pl_bmasse).toFixed(2)} M⊕` : "Нет данных"}</span></p>
              <p><span className="text-gray-400">Радиус:</span> <span className="text-white font-medium">{physicalParams.pl_rade ? `${parseFloat(physicalParams.pl_rade).toFixed(2)} R⊕` : "Нет данных"}</span></p>
              <p><span className="text-gray-400">Орбитальный период:</span> <span className="text-white font-medium">{orbitalParams.pl_orbper ? `${parseFloat(orbitalParams.pl_orbper).toFixed(2)} дней` : "Нет данных"}</span></p>
              <p><span className="text-gray-400">Расстояние до звезды:</span> <span className="text-white font-medium">{hostStar.distance ? `${parseFloat(hostStar.distance).toFixed(2)} пк` : "Нет данных"}</span></p>
            </div>
          </div>
        </div>
      </motion.div>

      {/* === Quick Stats Grid === */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <DataCard
          icon={Calendar}
          label="Год открытия"
          value={discoveryInfo.discovery_year}
          color="cyan"
        />
        <DataCard
          icon={Target}
          label="Орбитальный период"
          value={orbitalParams.pl_orbper ? `${parseFloat(orbitalParams.pl_orbper).toFixed(2)} дн.` : null}
          color="purple"
        />
        <DataCard
          icon={MapPin}
          label="Расстояние"
          value={hostStar.distance ? `${parseFloat(hostStar.distance).toFixed(2)} пк` : null}
          color="blue"
        />
        <DataCard
          icon={Info}
          label="Температура"
          value={physicalParams.pl_eqt ? `${parseFloat(physicalParams.pl_eqt).toFixed(0)} K` : null}
          color="green"
        />
      </div>

      {/* === Factors === */}
      {Object.keys(factors).length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="glass rounded-2xl p-6 border border-cyan-500/20"
        >
          <h3 className="text-xl font-semibold text-white mb-4 flex items-center gap-3">
            <BarChart2 className="w-5 h-5 text-cyan-400" />
            Ключевые факторы
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {Object.entries(factors).map(([key, value]) => (
              <Card key={key} className="bg-black/30 p-4 rounded-xl border border-cyan-500/10">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm font-medium text-cyan-300 capitalize">{key}</span>
                  <Badge variant="secondary" className="bg-cyan-500/20 text-cyan-400 text-xs">
                    {value.score || 0} баллов
                  </Badge>
                </div>
                <p className="text-xs text-gray-400">{value.status || "Неизвестно"}</p>
              </Card>
            ))}
          </div>
        </motion.div>
      )}

      {/* === Recommendations & Risks === */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {recommendations.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
            className="glass rounded-2xl p-6 border border-green-500/20"
          >
            <h3 className="text-xl font-semibold text-white mb-4 flex items-center gap-3">
              <CheckCircle className="w-5 h-5 text-green-400" />
              Рекомендации
            </h3>
            <ul className="space-y-3">
              {recommendations.map((rec, i) => (
                <li key={i} className="flex items-start gap-2 text-sm text-gray-300">
                  <CheckCircle className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
                  {rec}
                </li>
              ))}
            </ul>
          </motion.div>
        )}

        {risks.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.4 }}
            className="glass rounded-2xl p-6 border border-red-500/20"
          >
            <h3 className="text-xl font-semibold text-white mb-4 flex items-center gap-3">
              <AlertCircle className="w-5 h-5 text-red-400" />
              Риски
            </h3>
            <ul className="space-y-3">
              {risks.map((risk, i) => (
                <li key={i} className="flex items-start gap-2 text-sm text-gray-300">
                  <AlertCircle className="w-4 h-4 text-red-400 mt-0.5 flex-shrink-0" />
                  {risk}
                </li>
              ))}
            </ul>
          </motion.div>
        )}
      </div>

      {/* === Data Sources === */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
        className="glass rounded-xl p-6 border border-cyan-500/20"
      >
        <h3 className="text-xl font-semibold text-white mb-4 flex items-center gap-3">
          <Globe className="w-5 h-5 text-cyan-400" />
          Источники данных
        </h3>
        <div className="flex flex-wrap gap-3">
          {data.sources_used?.length > 0 ? (
            data.sources_used.map((source, i) => {
              const colorMap = {
                "NASA Exoplanet Archive": "bg-blue-500/20 border-blue-500/30 text-blue-300",
                "SIMBAD": "bg-purple-500/20 border-purple-500/30 text-purple-300",
                "Exoplanet.eu": "bg-green-500/20 border-green-500/30 text-green-300",
              };
              return (
                <div key={i} className={`px-4 py-2 border rounded-full text-sm ${colorMap[source] || "bg-gray-500/20 border-gray-500/30 text-gray-300"}`}>
                  {source}
                </div>
              );
            })
          ) : (
            <p className="text-gray-400">Данные из 1 источника</p>
          )}
        </div>
        <p className="text-sm text-gray-400 mt-3">
          Данные агрегированы из {data.sources_used?.length || 1} источника(ов)
        </p>
      </motion.div>
    </div>
  );
};

export default OverviewTab;