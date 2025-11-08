import { useState, useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { ArrowLeft, Download, Sparkles, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { motion } from "framer-motion";
import axios from "axios";
import { toast } from "sonner";
import OverviewTab from "@/components/tabs/OverviewTab";
import PhysicalTab from "@/components/tabs/PhysicalTab";
import OrbitalTab from "@/components/tabs/OrbitalTab";
import HostStarTab from "@/components/tabs/HostStarTab";
import ImagesTab from "@/components/tabs/ImagesTab";
import GraphsTab from "@/components/tabs/GraphsTab";
import AIAnalysisTab from "@/components/tabs/AIAnalysisTab";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const ResultsPage = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [data, setData] = useState(location.state?.data || null);
  const [activeTab, setActiveTab] = useState("overview");
  const [aiAnalysis, setAiAnalysis] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  useEffect(() => {
    if (!data) {
      navigate("/");
    }
  }, [data, navigate]);

  const handleAIAnalysis = async () => {
    setIsAnalyzing(true);
    try {
      const response = await axios.post(`${API}/analyze`, {
        exoplanet_data: data,
        ollama_url: "http://localhost:11434",
      });

      if (response.data.ollama_available) {
        setAiAnalysis(response.data.analysis);
        toast.success("AI анализ завершен");
      } else {
        toast.warning("Ollama недоступна. AI анализ пропущен.");
        setAiAnalysis(null);
      }
    } catch (error) {
      console.error("AI analysis error:", error);
      toast.error("Ошибка AI анализа");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleDownloadData = () => {
    const jsonData = JSON.stringify(data, null, 2);
    const blob = new Blob([jsonData], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${data.name}_data.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    toast.success("Данные загружены");
  };

  if (!data) {
    return null;
  }

  return (
    <div className="min-h-screen pb-12">
      {/* Header */}
      <div className="glass border-b border-cyan-500/20 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Button
                data-testid="back-button"
                onClick={() => navigate("/")}
                variant="ghost"
                size="sm"
                className="text-cyan-400 hover:text-cyan-300 hover:bg-cyan-500/10"
              >
                <ArrowLeft className="w-4 h-4 mr-2" />
                Назад
              </Button>
              <div>
                <h1 className="text-2xl sm:text-3xl font-bold text-white">
                  {data.name}
                </h1>
                {data.tic_id && (
                  <p className="text-sm text-gray-400">TIC ID: {data.tic_id}</p>
                )}
              </div>
            </div>
            <div className="flex gap-2">
              <Button
                data-testid="ai-analyze-button"
                onClick={handleAIAnalysis}
                disabled={isAnalyzing}
                size="sm"
                className="bg-purple-600 hover:bg-purple-500 text-white"
              >
                {isAnalyzing ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Анализ...
                  </>
                ) : (
                  <>
                    <Sparkles className="w-4 h-4 mr-2" />
                    AI Анализ
                  </>
                )}
              </Button>
              <Button
                data-testid="download-button"
                onClick={handleDownloadData}
                size="sm"
                variant="outline"
                className="border-cyan-500/30 text-cyan-400 hover:bg-cyan-500/10"
              >
                <Download className="w-4 h-4 mr-2" />
                Скачать
              </Button>
            </div>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-7xl mx-auto px-4 py-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
            <TabsList className="glass w-full justify-start overflow-x-auto flex-wrap h-auto p-2 gap-2 mb-6">
              <TabsTrigger
                data-testid="tab-overview"
                value="overview"
                className="data-[state=active]:bg-cyan-500/20 data-[state=active]:text-cyan-300"
              >
                Обзор
              </TabsTrigger>
              <TabsTrigger
                data-testid="tab-physical"
                value="physical"
                className="data-[state=active]:bg-cyan-500/20 data-[state=active]:text-cyan-300"
              >
                Физические параметры
              </TabsTrigger>
              <TabsTrigger
                data-testid="tab-orbital"
                value="orbital"
                className="data-[state=active]:bg-cyan-500/20 data-[state=active]:text-cyan-300"
              >
                Орбитальные параметры
              </TabsTrigger>
              <TabsTrigger
                data-testid="tab-host-star"
                value="host-star"
                className="data-[state=active]:bg-cyan-500/20 data-[state=active]:text-cyan-300"
              >
                Звезда-хозяин
              </TabsTrigger>
              <TabsTrigger
                data-testid="tab-images"
                value="images"
                className="data-[state=active]:bg-cyan-500/20 data-[state=active]:text-cyan-300"
              >
                Изображения
              </TabsTrigger>
              <TabsTrigger
                data-testid="tab-graphs"
                value="graphs"
                className="data-[state=active]:bg-cyan-500/20 data-[state=active]:text-cyan-300"
              >
                Графики и симуляции
              </TabsTrigger>
              <TabsTrigger
                data-testid="tab-ai"
                value="ai"
                className="data-[state=active]:bg-cyan-500/20 data-[state=active]:text-cyan-300"
              >
                AI Анализ
              </TabsTrigger>
            </TabsList>

            <TabsContent value="overview">
              <OverviewTab data={data} />
            </TabsContent>

            <TabsContent value="physical">
              <PhysicalTab data={data.physical_params} />
            </TabsContent>

            <TabsContent value="orbital">
              <OrbitalTab data={data.orbital_params} />
            </TabsContent>

            <TabsContent value="host-star">
              <HostStarTab data={data.host_star} />
            </TabsContent>

            <TabsContent value="images">
              <ImagesTab
                syntheticImage={data.visualizations?.synthetic_star}
                planetName={data.name}
                hostStar={data.host_star?.name}
              />
            </TabsContent>

            <TabsContent value="graphs">
              <GraphsTab visualizations={data.visualizations} />
            </TabsContent>

            <TabsContent value="ai">
              <AIAnalysisTab
                analysis={aiAnalysis}
                isAnalyzing={isAnalyzing}
                onAnalyze={handleAIAnalysis}
              />



            
            </TabsContent>
          </Tabs>
        </motion.div>
      </div>
    </div>
  );
};

export default ResultsPage;
