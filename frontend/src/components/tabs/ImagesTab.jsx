// src/components/tabs/ImagesTab.jsx
import { motion } from "framer-motion";
import { Image as ImageIcon, Star, Download, Globe, Telescope, Satellite, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";

const ImagesTab = ({ syntheticImage, planetName, hostStar, realImages = [] }) => {
  const handleDownload = (imageUrl, filename) => {
    const link = document.createElement('a');
    link.href = imageUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const getSourceIcon = (source) => {
    if (source.includes("SkyView") || source.includes("2MASS")) return <Globe className="w-4 h-4" />;
    if (source.includes("Hubble")) return <Telescope className="w-4 h-4" />;
    if (source.includes("JWST")) return <Satellite className="w-4 h-4" />;
    return <ImageIcon className="w-4 h-4" />;
  };

  const getSourceColor = (source) => {
    if (source.includes("SkyView") || source.includes("2MASS")) return "border-cyan-500/30";
    if (source.includes("Hubble")) return "border-purple-500/30";
    if (source.includes("JWST")) return "border-green-500/30";
    return "border-gray-500/30";
  };

  return (
    <div className="space-y-6">
      {/* 1. Synthetic Star Image */}
      {syntheticImage && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="glass rounded-2xl p-6 border border-yellow-500/20"
        >
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <Star className="w-6 h-6 text-yellow-400" />
              <h2 className="text-2xl font-bold text-white">Синтетическое изображение звезды</h2>
            </div>
            <Button
              size="sm"
              variant="outline"
              className="border-yellow-500/30 text-yellow-400 hover:bg-yellow-500/10"
              onClick={() => handleDownload(syntheticImage, `${hostStar || 'star'}_synthetic.png`)}
            >
              <Download className="w-4 h-4 mr-2" />
              Скачать
            </Button>
          </div>
          <div className="flex justify-center">
            <div className="relative rounded-xl overflow-hidden glow-strong">
              <img
                src={syntheticImage}
                alt="Synthetic star"
                className="max-w-full h-auto rounded-xl shadow-2xl"
                style={{ maxHeight: '500px' }}
              />
            </div>
          </div>
          <p className="text-gray-400 text-center mt-4">
            Генерация на основе температуры и радиуса звезды <strong>{hostStar}</strong>
          </p>
        </motion.div>
      )}

      {/* 2. Real Images Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.2 }}
        className="space-y-4"
      >
        <div className="flex items-center gap-3 mb-6">
          <ImageIcon className="w-6 h-6 text-cyan-400" />
          <h2 className="text-2xl font-bold text-white">
            Реальные изображения ({realImages.length > 0 ? realImages.length : "0"})
          </h2>
        </div>

        {realImages.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            {realImages.map((img, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: index * 0.1 }}
              >
                <Card className={`glass border ${getSourceColor(img.source)} overflow-hidden`}>
                  <div className="p-4 border-b border-cyan-500/10 flex items-center justify-between bg-black/20">
                    <div className="flex items-center gap-2">
                      {getSourceIcon(img.source)}
                      <span className="text-sm font-medium text-cyan-300">{img.source}</span>
                    </div>
                    <Button
                      size="sm"
                      variant="ghost"
                      className="text-cyan-400 hover:text-cyan-300"
                      onClick={() => handleDownload(img.url, `${planetName}_${img.source.replace(/[^a-z0-9]/gi, '_')}.png`)}
                    >
                      <Download className="w-4 h-4" />
                    </Button>
                  </div>
                  <div className="p-4 bg-gradient-to-b from-black/40 to-black/20">
                    <div className="relative rounded-lg overflow-hidden glow">
                      <img
                        src={img.url}
                        alt={img.source}
                        className="w-full h-auto rounded-lg shadow-xl"
                        style={{ maxHeight: '400px', objectFit: 'contain' }}
                        loading="lazy"
                      />
                    </div>
                  </div>
                  <div className="p-3 text-xs text-gray-400 border-t border-cyan-500/10">
                    {img.description || "Изображение звезды-хозяина из архива"}
                  </div>
                </Card>
              </motion.div>
            ))}
          </div>
        ) : (
          <Alert className="glass border border-orange-500/30 bg-orange-500/5">
            <AlertCircle className="h-4 w-4 text-orange-400" />
            <AlertDescription className="text-orange-300">
              Реальные изображения звезды <strong>{hostStar}</strong> не найдены в NASA SkyView, Hubble или JWST.  
              Это нормально — большинство звёзд слишком тусклые или далеко. Доступно только синтетическое изображение.
            </AlertDescription>
          </Alert>
        )}
      </motion.div>

      {/* 3. Info Block */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
        className="glass rounded-xl p-6 border border-blue-500/20"
      >
        <h3 className="text-lg font-semibold text-white mb-3">Почему нет фото планеты?</h3>
        <p className="text-gray-400 text-sm leading-relaxed">
          Прямые изображения экзопланет — редкость. На расстоянии в сотни световых лет планета в миллиарды раз тусклее звезды.  
          Мы показываем <strong>реальные снимки звезды-хозяина</strong> из NASA SkyView (инфракрасные), Hubble и JWST.  
          Если изображение не загружено — звезда слишком слабая или координаты неточны.
        </p>
      </motion.div>
    </div>
  );
};

export default ImagesTab;