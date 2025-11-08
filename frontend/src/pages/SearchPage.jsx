import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Search, Sparkles, Globe2, Telescope } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { motion } from "framer-motion";
import axios from "axios";
import { toast } from "sonner";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const SearchPage = () => {
  const [query, setQuery] = useState("");
  const [searchType, setSearchType] = useState("name");
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();

  const handleSearch = async () => {
    const searchQuery = searchType === "tic" ? `TIC ${query.trim()}` : query.trim();
    
    if (!query.trim()) {
      toast.error("Пожалуйста, введите название или TIC ID");
      return;
    }

    setIsLoading(true);
    try {
      const response = await axios.post(`${API}/search`, {
        query: searchQuery,
        search_type: searchType,
      });

      if (response.data) {
        // Navigate to results page with data
        navigate("/results", { state: { data: response.data } });
      }
    } catch (error) {
      console.error("Search error:", error);
      if (error.response?.status === 404) {
        toast.error("Данные не найдены. Попробуйте другой запрос.");
      } else {
        toast.error("Ошибка поиска. Попробуйте снова.");
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter") {
      handleSearch();
    }
  };

  const handleSearchTypeChange = (newType) => {
    setSearchType(newType);
    setQuery(""); // Clear query when switching types
  };

  const handleQueryChange = (e) => {
    const value = e.target.value;
    if (searchType === "tic") {
      // Only allow numbers for TIC search
      const numericValue = value.replace(/[^0-9]/g, "");
      setQuery(numericValue);
    } else {
      setQuery(value);
    }
  };

  return (
    <div className="min-h-screen relative overflow-hidden">
      {/* Animated background elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <motion.div
          className="absolute top-20 left-10 w-64 h-64 rounded-full"
          style={{
            background: "radial-gradient(circle, rgba(0,255,255,0.1) 0%, transparent 70%)",
          }}
          animate={{
            scale: [1, 1.2, 1],
            opacity: [0.3, 0.5, 0.3],
          }}
          transition={{
            duration: 4,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        />
        <motion.div
          className="absolute bottom-20 right-10 w-96 h-96 rounded-full"
          style={{
            background: "radial-gradient(circle, rgba(100,150,255,0.1) 0%, transparent 70%)",
          }}
          animate={{
            scale: [1, 1.3, 1],
            opacity: [0.2, 0.4, 0.2],
          }}
          transition={{
            duration: 5,
            repeat: Infinity,
            ease: "easeInOut",
            delay: 1,
          }}
        />
        <motion.div
          className="absolute top-1/2 left-1/2 w-80 h-80 rounded-full"
          style={{
            background: "radial-gradient(circle, rgba(150,100,255,0.08) 0%, transparent 70%)",
            transform: "translate(-50%, -50%)",
          }}
          animate={{
            scale: [1, 1.1, 1],
            rotate: [0, 180, 360],
          }}
          transition={{
            duration: 8,
            repeat: Infinity,
            ease: "linear",
          }}
        />
      </div>

      {/* Main content */}
      <div className="relative z-10 flex flex-col items-center justify-center min-h-screen px-4">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-center mb-12"
        >
          <motion.div
            className="inline-block mb-6"
            animate={{ rotate: [0, 5, -5, 0] }}
            transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
          >
            <Telescope className="w-20 h-20 text-cyan-400" strokeWidth={1.5} />
          </motion.div>
          
          <h1 className="text-5xl sm:text-6xl lg:text-7xl font-bold mb-4">
            <span className="bg-gradient-to-r from-cyan-400 via-blue-400 to-purple-400 bg-clip-text text-transparent">
              ExoPlanet
            </span>
            <br />
            <span className="text-white">Explorer</span>
          </h1>
          
          <p className="text-lg sm:text-xl text-gray-300 max-w-2xl mx-auto">
            Исследуйте тысячи экзопланет из всех доступных источников данных
          </p>
        </motion.div>

        {/* Search box */}
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="w-full max-w-3xl"
        >
          <div className="glass rounded-2xl p-8 glow">
            <div className="flex flex-col gap-4">
              {/* Search type selector */}
              <div className="flex items-center gap-3">
                <label className="text-sm font-medium text-gray-300 min-w-[120px]">
                  Тип поиска:
                </label>
                <Select value={searchType} onValueChange={handleSearchTypeChange}>
                  <SelectTrigger data-testid="search-type-select" className="w-full bg-slate-800/50 border-cyan-500/30 text-white focus:border-cyan-400">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-slate-800 border-cyan-500/30">
                    <SelectItem data-testid="search-type-name" value="name">По названию</SelectItem>
                    <SelectItem data-testid="search-type-tic" value="tic">По TIC ID</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Search input */}
              <div className="relative">
                {searchType === "tic" && (
                  <span className="absolute left-14 top-1/2 transform -translate-y-1/2 text-lg text-cyan-400 font-semibold pointer-events-none">
                    TIC
                  </span>
                )}
                <Input
                  data-testid="search-input"
                  type="text"
                  placeholder={searchType === "name" ? "Введите название (например, Kepler-186 f)" : "Введите только цифры (например, 268159881)"}
                  value={query}
                  onChange={handleQueryChange}
                  onKeyPress={handleKeyPress}
                  className={`w-full h-14 ${searchType === "tic" ? "pl-24" : "pl-14"} pr-4 text-lg bg-slate-800/50 border-cyan-500/30 text-white placeholder-gray-400 focus:border-cyan-400 rounded-xl`}
                />
                <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 w-6 h-6 text-cyan-400" />
              </div>

              {/* Search button */}
              <Button
                data-testid="search-button"
                onClick={handleSearch}
                disabled={isLoading}
                className="h-14 text-lg font-semibold bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white rounded-xl shadow-lg hover:shadow-cyan-500/50"
              >
                {isLoading ? (
                  <>
                    <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
                    Поиск...
                  </>
                ) : (
                  <>
                    <Sparkles className="w-5 h-5 mr-2" />
                    Найти экзопланету
                  </>
                )}
              </Button>
            </div>
          </div>
        </motion.div>

        {/* Popular searches */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="mt-8 text-center"
        >
          <p className="text-sm text-gray-400 mb-3">Популярные запросы:</p>
          <div className="flex flex-wrap gap-2 justify-center">
            {["Kepler-186 f", "TRAPPIST-1 e", "Barnard b", "HD 209458 b"].map(
              (example) => (
                <button
                  key={example}
                  onClick={() => setQuery(example)}
                  className="px-4 py-2 text-sm bg-slate-800/50 hover:bg-slate-700/50 border border-cyan-500/20 hover:border-cyan-400/50 rounded-full text-cyan-300 hover:text-cyan-200"
                  data-testid={`example-${example.replace(/\s/g, '-').toLowerCase()}`}
                >
                  {example}
                </button>
              )
            )}
          </div>
        </motion.div>

        {/* Features */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1, delay: 0.6 }}
          className="mt-16 grid grid-cols-1 sm:grid-cols-3 gap-6 max-w-4xl"
        >
          <div className="glass rounded-xl p-6 text-center hover:scale-105 transition-transform">
            <Globe2 className="w-12 h-12 mx-auto mb-3 text-cyan-400" />
            <h3 className="font-semibold text-white mb-2">Множество источников</h3>
            <p className="text-sm text-gray-400">
              NASA, SIMBAD, Exoplanet.eu
            </p>
          </div>
          <div className="glass rounded-xl p-6 text-center hover:scale-105 transition-transform">
            <Sparkles className="w-12 h-12 mx-auto mb-3 text-purple-400" />
            <h3 className="font-semibold text-white mb-2">AI анализ</h3>
            <p className="text-sm text-gray-400">
              Интеллектуальная обработка данных
            </p>
          </div>
          <div className="glass rounded-xl p-6 text-center hover:scale-105 transition-transform">
            <Telescope className="w-12 h-12 mx-auto mb-3 text-blue-400" />
            <h3 className="font-semibold text-white mb-2">Визуализация</h3>
            <p className="text-sm text-gray-400">
              Графики, симуляции, изображения
            </p>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default SearchPage;
