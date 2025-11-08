import { motion } from "framer-motion";
import { Orbit, Clock, Compass, Route } from "lucide-react";

const OrbitalParam = ({ icon: Icon, label, value, unit }) => {
  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.4 }}
      className="glass rounded-xl p-6 hover:scale-105 transition-transform"
    >
      <div className="flex items-center gap-3 mb-3">
        <div className="p-2 bg-cyan-500/20 rounded-lg">
          <Icon className="w-5 h-5 text-cyan-400" />
        </div>
        <h3 className="font-semibold text-white">{label}</h3>
      </div>
      <p className="text-2xl font-bold text-white">
        {value !== null && value !== undefined ? value : "—"}
      </p>
      {unit && <p className="text-sm text-gray-400 mt-1">{unit}</p>}
    </motion.div>
  );
};

const OrbitalTab = ({ data }) => {
  if (!data || Object.keys(data).length === 0) {
    return (
      <div className="glass rounded-2xl p-12 text-center">
        <p className="text-xl text-gray-400">Орбитальные параметры не найдены</p>
      </div>
    );
  }

  const period = data.period || data.period_eu;
  const semiMajorAxis = data.semi_major_axis || data.semi_major_axis_eu;
  const eccentricity = data.eccentricity || data.eccentricity_eu;
  const inclination = data.inclination;

  return (
    <div className="space-y-6">
      {/* Main orbital parameters */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <OrbitalParam
          icon={Clock}
          label="Орбитальный период"
          value={period ? period.toFixed(2) : null}
          unit="дней"
        />
        <OrbitalParam
          icon={Route}
          label="Большая полуось"
          value={semiMajorAxis ? semiMajorAxis.toFixed(4) : null}
          unit="а.е. (AU)"
        />
        <OrbitalParam
          icon={Orbit}
          label="Эксцентриситет"
          value={eccentricity ? eccentricity.toFixed(4) : null}
          unit="(0 = круг, 1 = эллипс)"
        />
        <OrbitalParam
          icon={Compass}
          label="Наклонение орбиты"
          value={inclination ? inclination.toFixed(2) : null}
          unit="градусы"
        />
      </div>

      {/* Detailed information */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="glass rounded-xl p-6"
      >
        <h3 className="text-2xl font-bold text-white mb-4">Описание орбиты</h3>
        <div className="space-y-4 text-gray-300">
          {period && (
            <div>
              <h4 className="font-semibold text-cyan-300 mb-2">Орбитальный период</h4>
              <p>
                Планета совершает полный оборот вокруг своей звезды за <span className="text-white font-semibold">{period.toFixed(2)} земных дней</span>.
                {period < 1 && " Это очень короткий период - планета очень близко к своей звезде."}
                {period >= 1 && period < 365 && " Это меньше земного года."}
                {period >= 365 && period < 4000 && " Примерно как у планет внутренней Солнечной системы."}
                {period >= 4000 && " Очень длинный период, как у внешних планет."}
              </p>
            </div>
          )}

          {semiMajorAxis && (
            <div>
              <h4 className="font-semibold text-cyan-300 mb-2">Расстояние до звезды</h4>
              <p>
                Среднее расстояние от планеты до звезды: <span className="text-white font-semibold">{semiMajorAxis.toFixed(4)} а.е.</span>
                {semiMajorAxis < 0.1 && " - очень близко к звезде (горячий Юпитер)"}
                {semiMajorAxis >= 0.1 && semiMajorAxis < 2 && " - в пределах внутренней Солнечной системы"}
                {semiMajorAxis >= 2 && semiMajorAxis < 30 && " - как у внешних планет"}
                {semiMajorAxis >= 30 && " - очень далеко от звезды"}
                . Для сравнения: Земля находится на расстоянии 1 а.е. от Солнца.
              </p>
            </div>
          )}

          {eccentricity !== null && eccentricity !== undefined && (
            <div>
              <h4 className="font-semibold text-cyan-300 mb-2">Эксцентриситет орбиты</h4>
              <p>
                Эксцентриситет <span className="text-white font-semibold">{eccentricity.toFixed(4)}</span>
                {eccentricity < 0.05 && " означает почти круговую орбиту - расстояние до звезды почти постоянное."}
                {eccentricity >= 0.05 && eccentricity < 0.2 && " означает слегка эллиптическую орбиту."}
                {eccentricity >= 0.2 && eccentricity < 0.5 && " означает заметно эллиптическую орбиту - расстояние до звезды значительно изменяется."}
                {eccentricity >= 0.5 && " означает очень вытянутую орбиту - планета то приближается, то удаляется от звезды."}
              </p>
            </div>
          )}

          {inclination && (
            <div>
              <h4 className="font-semibold text-cyan-300 mb-2">Наклонение орбиты</h4>
              <p>
                Орбита наклонена на <span className="text-white font-semibold">{inclination.toFixed(2)}°</span> относительно линии зрения с Земли.
                {inclination > 85 && " Почти перпендикулярно к нашему обзору - мы наблюдаем транзиты."}
              </p>
            </div>
          )}
        </div>
      </motion.div>
    </div>
  );
};

export default OrbitalTab;
