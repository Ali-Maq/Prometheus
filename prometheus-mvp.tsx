import React, { useState, useEffect } from 'react';
import { ChevronRight, ChevronDown, Search, FileText, Activity, Dna, Brain, Star, ExternalLink, Download, MessageCircle, AlertCircle, CheckCircle, Info } from 'lucide-react';

// Mock Data and Knowledge Bases
const mockPatientData = {
  id: "MM-001",
  name: "Patient Demo",
  age: 62,
  sex: "Male",
  diagnosis: "Multiple Myeloma",
  stage: "R-ISS III",
  priorRegimens: [
    { name: "VRd", response: "VGPR", duration: "18 months" },
    { name: "KRd", response: "PR", duration: "12 months" },
    { name: "Dara-Pd", response: "PD", duration: "6 months" }
  ],
  genomicVariants: [
    { gene: "BRAF", variant: "V600E", vaf: 0.42, type: "SNV", chromosome: "7", position: "140753336" },
    { gene: "TP53", variant: "R175H", vaf: 0.38, type: "SNV", chromosome: "17", position: "7674230" },
    { gene: "KRAS", variant: "G12D", vaf: 0.31, type: "SNV", chromosome: "12", position: "25245350" },
    { gene: "CDKN2C", variant: "Deletion", vaf: 0.0, type: "CNV", chromosome: "1", position: "51006000" }
  ],
  rnaExpression: {
    BCL2: { tpm: 45.2, percentile: 85, status: "High" },
    MYC: { tpm: 123.4, percentile: 92, status: "High" },
    FGFR3: { tpm: 12.1, percentile: 25, status: "Low" },
    XPO1: { tpm: 78.9, percentile: 75, status: "High" }
  },
  drugSignatures: {
    venetoclax: { score: 0.78, evidence: "BCL2-high signature" },
    selinexor: { score: 0.65, evidence: "XPO1-high signature" },
    dabrafenib: { score: 0.82, evidence: "BRAF V600E mutation" }
  }
};

// ML Model Simulation
const predictResponseProbability = (variants, expression, signatures) => {
  let score = 0.5;
  
  variants.forEach(variant => {
    if (variant.gene === "BRAF" && variant.variant === "V600E") score += 0.25;
    if (variant.gene === "TP53") score -= 0.15;
    if (variant.type === "CNV" && variant.gene === "CDKN2C") score += 0.1;
  });
  
  if (expression.BCL2 && expression.BCL2.status === "High") score += 0.2;
  if (expression.MYC && expression.MYC.status === "High") score -= 0.1;
  
  return Math.min(Math.max(score, 0), 1);
};

// Patient Snapshot Component
const PatientSnapshot = ({ patient }) => (
  <div className="bg-white rounded-lg border p-6 mb-6">
    <h2 className="text-xl font-bold mb-4 flex items-center">
      <FileText className="w-5 h-5 mr-2" />
      Patient Snapshot
    </h2>
    <div className="grid grid-cols-2 gap-4">
      <div>
        <p><span className="font-semibold">ID:</span> {patient.id}</p>
        <p><span className="font-semibold">Age/Sex:</span> {patient.age}yr {patient.sex}</p>
        <p><span className="font-semibold">Diagnosis:</span> {patient.diagnosis}</p>
        <p><span className="font-semibold">Stage:</span> {patient.stage}</p>
      </div>
      <div>
        <p className="font-semibold mb-2">Prior Regimens:</p>
        {patient.priorRegimens.map((regimen, idx) => (
          <div key={idx} className="text-sm">
            {regimen.name} → {regimen.response} ({regimen.duration})
          </div>
        ))}
      </div>
    </div>
  </div>
);

// Multi-Omic Viewer Component
const MultiOmicViewer = ({ variants, expression }) => (
  <div className="bg-white rounded-lg border p-6 mb-6">
    <h2 className="text-xl font-bold mb-4 flex items-center">
      <Dna className="w-5 h-5 mr-2" />
      Multi-Omic Profile
    </h2>
    
    <div className="mb-6">
      <h3 className="font-semibold mb-3">Genomic Variants</h3>
      <div className="space-y-2">
        {variants.map((variant, idx) => (
          <div key={idx} className="flex items-center justify-between p-3 bg-gray-50 rounded">
            <div>
              <span className="font-medium">{variant.gene}</span>
              <span className="mx-2 text-gray-600">{variant.variant}</span>
              <span className="text-sm text-gray-500">({variant.type})</span>
            </div>
            <div className="text-right">
              <div className="text-sm">VAF: {variant.vaf}</div>
              <div className="text-xs text-gray-500">{variant.chromosome}:{variant.position}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
    
    <div>
      <h3 className="font-semibold mb-3">Key Gene Expression</h3>
      <div className="grid grid-cols-2 gap-4">
        {Object.entries(expression).map(([gene, data]) => (
          <div key={gene} className="p-3 bg-gray-50 rounded">
            <div className="flex justify-between items-center">
              <span className="font-medium">{gene}</span>
              <span className={`px-2 py-1 rounded text-xs ${
                data.status === 'High' ? 'bg-red-100 text-red-800' : 
                data.status === 'Low' ? 'bg-blue-100 text-blue-800' : 
                'bg-gray-100 text-gray-800'
              }`}>
                {data.status}
              </span>
            </div>
            <div className="text-sm text-gray-600">
              TPM: {data.tpm} ({data.percentile}th percentile)
            </div>
          </div>
        ))}
      </div>
    </div>
  </div>
);

// Enhanced Reasoning Engine Component
const ReasoningEngine = ({ patient, onRecommendationsGenerated }) => {
  const [processing, setProcessing] = useState(false);
  const [step, setStep] = useState(0);
  const [recommendations, setRecommendations] = useState([]);
  const [expandedStep, setExpandedStep] = useState(null);
  const [agentLogs, setAgentLogs] = useState({});
  const [apiResponses, setApiResponses] = useState({});
  const [confidenceMetrics, setConfidenceMetrics] = useState({});
  
  const steps = [
    {
      name: "Data Ingestion & Validation",
      agents: ["ValidationAgent", "NormalizationAgent"],
      description: "Multi-agent data quality assurance and standardization"
    },
    {
      name: "Multi-KB Variant Mapping", 
      agents: ["OncoKBAgent", "CIViCAgent", "CGIAgent", "ConsensusAgent"],
      description: "Parallel knowledge base queries with consensus verification"
    },
    {
      name: "Literature RAG Processing",
      agents: ["SearchAgent", "RetrievalAgent", "SynthesisAgent"],
      description: "Disease-specific literature mining and relevance scoring"
    },
    {
      name: "Clinical Guidelines Engine",
      agents: ["NCCNAgent", "IWMGAgent", "GuidelineConsensusAgent"], 
      description: "Structured guideline interpretation and pathway mapping"
    },
    {
      name: "ML Ensemble Prediction",
      agents: ["XGBoostAgent", "RandomForestAgent", "NeuralNetAgent", "EnsembleAgent"],
      description: "Multi-model outcome prediction with uncertainty quantification"
    },
    {
      name: "Cross-Validation & Ranking",
      agents: ["ValidationAgent", "RankingAgent", "QualityAssuranceAgent"],
      description: "Multi-agent verification and confidence scoring"
    }
  ];
  
  useEffect(() => {
    generateRecommendations();
  }, []);
  
  const generateRecommendations = async () => {
    setProcessing(true);
    
    for (let i = 0; i < steps.length; i++) {
      setStep(i);
      await simulateStepProcessing(i);
      await new Promise(resolve => setTimeout(resolve, 1200));
    }
    
    const recs = await generateFinalRecommendations();
    setRecommendations(recs);
    setProcessing(false);
    onRecommendationsGenerated(recs);
  };
  
  const simulateStepProcessing = async (stepIndex) => {
    const logs = {};
    const apis = {};
    const confidence = {};
    
    switch(stepIndex) {
      case 0:
        logs.ValidationAgent = [
          "Validating genomic coordinates (GRCh38)",
          "VCF format compliance check",
          "VAF threshold validation (>0.05)",
          "Expression data QC (TPM normalization)"
        ];
        logs.NormalizationAgent = [
          "Converting HGVS nomenclature",
          "Standardizing gene symbols (HUGO)",
          "Mapping to Ensembl IDs",
          "Data harmonization complete"
        ];
        confidence.overall = 0.95;
        break;
        
      case 1:
        apis.OncoKB = {
          endpoint: "https://oncokb.org/api/v1/annotate/mutations/byProteinChange",
          requests: [
            { gene: "BRAF", alteration: "V600E", response: "Level 1 - FDA Approved" },
            { gene: "TP53", alteration: "R175H", response: "Level 3A - Clinical Evidence" }
          ]
        };
        logs.ConsensusAgent = [
          "Cross-referencing BRAF V600E across 3 KBs",
          "High consensus (3/3) for BRAF inhibitor sensitivity",
          "Validating TP53 R175H actionability",
          "Consensus scoring complete"
        ];
        confidence.consensus = 0.85;
        break;
        
      case 2:
        apis.PubMed = {
          query: "(BRAF V600E OR venetoclax) AND multiple myeloma",
          results: 1247,
          relevant: 89
        };
        logs.SearchAgent = [
          "Query: BRAF V600E multiple myeloma treatment",
          "Retrieved 1,247 abstracts",
          "Applying relevance filters",
          "89 high-relevance papers identified"
        ];
        confidence.literature = 0.78;
        break;
        
      case 3:
        apis.NCCN = {
          version: "v1.2025",
          pathways: ["Relapsed/Refractory MM"],
          recommendations: [
            { regimen: "Dara-KRd", category: "1" }
          ]
        };
        logs.NCCNAgent = [
          "Parsing NCCN MM guidelines v1.2025",
          "Patient stage: R-ISS III, 3+ prior lines",
          "12 guideline-concordant options identified"
        ];
        confidence.guidelines = 0.91;
        break;
        
      case 4:
        logs.XGBoostAgent = [
          "Feature engineering: 47 genomic + 23 clinical variables",
          "Model inference on patient profile",
          "Response probability: BRAF inh. 0.83"
        ];
        logs.EnsembleAgent = [
          "Weighted ensemble (XGB: 0.4, RF: 0.3, NN: 0.3)",
          "Final probability distributions generated"
        ];
        confidence.ml_ensemble = 0.82;
        break;
        
      case 5:
        logs.ValidationAgent = [
          "Cross-validating KB evidence with literature",
          "Checking drug-drug interaction conflicts",
          "Safety validation complete"
        ];
        logs.RankingAgent = [
          "Multi-criteria decision analysis",
          "Tiered recommendations generated"
        ];
        confidence.final_ranking = 0.87;
        break;
    }
    
    setAgentLogs(prev => ({ ...prev, [stepIndex]: logs }));
    setApiResponses(prev => ({ ...prev, [stepIndex]: apis }));
    setConfidenceMetrics(prev => ({ ...prev, [stepIndex]: confidence }));
  };
  
  const generateFinalRecommendations = async () => {
    return [
      {
        drug: "Dabrafenib + Trametinib",
        tier: "1",
        confidence: 0.87,
        rationale: "BRAF V600E mutation with cross-validated KB evidence and high ML prediction score",
        evidence: [
          { source: "OncoKB", level: "1", description: "FDA-approved for BRAF V600E+ tumors" },
          { source: "Literature", pmid: "34567890", description: "Case series in BRAF+ myeloma" },
          { source: "ML Ensemble", description: "83% response probability" }
        ],
        responseProb: 0.83,
        mechanism: "BRAF/MEK pathway inhibition",
        trials: ["NCT02034110"],
        agentConsensus: {
          oncoKB: "Recommend",
          civic: "Recommend", 
          guidelines: "Off-guideline",
          ml: "High confidence"
        }
      },
      {
        drug: "Venetoclax + Dexamethasone", 
        tier: "2A",
        confidence: 0.74,
        rationale: "BCL2 overexpression with venetoclax sensitivity signature",
        evidence: [
          { source: "Expression Analysis", description: "BCL2 high expression" },
          { source: "Drug Signature", description: "Venetoclax sensitivity score: 0.78" }
        ],
        responseProb: 0.67,
        mechanism: "BCL2 apoptosis pathway",
        trials: ["NCT03567616"],
        agentConsensus: {
          expression: "Strong signal",
          literature: "Supportive",
          guidelines: "2B evidence",
          ml: "Moderate confidence"
        }
      }
    ];
  };
  
  const toggleStepExpansion = (stepIndex) => {
    setExpandedStep(expandedStep === stepIndex ? null : stepIndex);
  };
  
  return (
    <div className="bg-white rounded-lg border p-6 mb-6">
      <h2 className="text-xl font-bold mb-4 flex items-center">
        <Brain className="w-5 h-5 mr-2" />
        Multi-Agent Reasoning Engine
      </h2>
      
      {processing ? (
        <div className="space-y-3">
          {steps.map((stepData, idx) => (
            <div key={idx} className={`border rounded-lg ${
              idx < step ? 'bg-green-50 border-green-200' :
              idx === step ? 'bg-blue-50 border-blue-200' :
              'bg-gray-50 border-gray-200'
            }`}>
              <div 
                className="flex items-center justify-between p-4 cursor-pointer"
                onClick={() => toggleStepExpansion(idx)}
              >
                <div className="flex items-center">
                  {idx < step ? <CheckCircle className="w-5 h-5 mr-3 text-green-600" /> :
                   idx === step ? <Activity className="w-5 h-5 mr-3 animate-spin text-blue-600" /> :
                   <div className="w-5 h-5 mr-3 rounded-full border-2 border-gray-300" />}
                  <div>
                    <div className="font-semibold">{stepData.name}</div>
                    <div className="text-sm text-gray-600">{stepData.description}</div>
                    <div className="text-xs text-gray-500 mt-1">
                      Agents: {stepData.agents.join(", ")}
                    </div>
                  </div>
                </div>
                <div className="flex items-center">
                  {confidenceMetrics[idx] && (
                    <div className="text-right mr-4">
                      <div className="text-sm font-medium">
                        Confidence: {Math.round(Object.values(confidenceMetrics[idx])[0] * 100)}%
                      </div>
                    </div>
                  )}
                  {expandedStep === idx ? <ChevronDown className="w-5 h-5" /> : <ChevronRight className="w-5 h-5" />}
                </div>
              </div>
              
              {expandedStep === idx && agentLogs[idx] && (
                <div className="border-t bg-white p-4">
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <div>
                      <h4 className="font-semibold mb-3 text-gray-900">Agent Activity Logs</h4>
                      {Object.entries(agentLogs[idx]).map(([agent, logs]) => (
                        <div key={agent} className="mb-4 p-3 bg-gray-50 rounded">
                          <div className="font-medium text-sm text-blue-700 mb-2">{agent}</div>
                          {logs.map((log, logIdx) => (
                            <div key={logIdx} className="text-xs font-mono text-gray-700 mb-1">
                              {log}
                            </div>
                          ))}
                        </div>
                      ))}
                    </div>
                    
                    <div>
                      <h4 className="font-semibold mb-3 text-gray-900">API Calls & Metrics</h4>
                      {apiResponses[idx] && Object.entries(apiResponses[idx]).map(([api, data]) => (
                        <div key={api} className="mb-4 p-3 bg-blue-50 rounded">
                          <div className="font-medium text-sm text-blue-700 mb-2">{api} API</div>
                          {data.endpoint && (
                            <div className="text-xs text-gray-600 mb-2">
                              <span className="font-medium">Endpoint:</span> {data.endpoint}
                            </div>
                          )}
                          {data.requests && (
                            <div className="space-y-1">
                              {data.requests.slice(0, 2).map((req, reqIdx) => (
                                <div key={reqIdx} className="text-xs bg-white p-2 rounded border">
                                  {req.gene}: {req.response}
                                </div>
                              ))}
                            </div>
                          )}
                          {data.query && (
                            <div className="text-xs text-gray-700">
                              <div><span className="font-medium">Query:</span> {data.query}</div>
                              <div><span className="font-medium">Results:</span> {data.results} total, {data.relevant} relevant</div>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      ) : (
        <div className="space-y-4">
          <div className="text-green-600 flex items-center">
            <CheckCircle className="w-5 h-5 mr-2" />
            Multi-agent analysis complete - {recommendations.length} recommendations generated
          </div>
          
          {recommendations.length > 0 && (
            <div className="p-4 bg-gray-50 rounded-lg">
              <h3 className="font-semibold mb-3">Agent Consensus Summary</h3>
              <div className="grid grid-cols-3 gap-4 text-sm">
                <div>
                  <div className="font-medium text-gray-700">Knowledge Base Agents</div>
                  <div className="text-green-600">High consensus on BRAF targeting</div>
                </div>
                <div>
                  <div className="font-medium text-gray-700">ML Ensemble</div>
                  <div className="text-green-600">83% confidence on BRAF response</div>
                </div>
                <div>
                  <div className="font-medium text-gray-700">Guidelines Compliance</div>
                  <div className="text-orange-600">BRAF inh: Off-guideline</div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// Top Recommendations Component
const TopRecommendations = ({ recommendations }) => {
  const [selectedRec, setSelectedRec] = useState(null);
  const [showAgentDetails, setShowAgentDetails] = useState({});
  
  const getTierColor = (tier) => {
    switch(tier) {
      case "1": return "bg-green-100 text-green-800";
      case "2A": return "bg-yellow-100 text-yellow-800"; 
      case "2B": return "bg-orange-100 text-orange-800";
      default: return "bg-gray-100 text-gray-800";
    }
  };
  
  const getConsensusColor = (consensus) => {
    switch(consensus) {
      case "Recommend": case "High confidence": case "Strong signal":
        return "text-green-700 bg-green-50";
      case "Moderate confidence": case "Supportive":
        return "text-yellow-700 bg-yellow-50";
      case "Off-guideline": case "2B evidence":
        return "text-orange-700 bg-orange-50";
      default:
        return "text-gray-700 bg-gray-50";
    }
  };
  
  const toggleAgentDetails = (recIdx) => {
    setShowAgentDetails(prev => ({
      ...prev,
      [recIdx]: !prev[recIdx]
    }));
  };
  
  return (
    <div className="bg-white rounded-lg border p-6 mb-6">
      <h2 className="text-xl font-bold mb-4 flex items-center">
        <Star className="w-5 h-5 mr-2" />
        Top Recommendations
      </h2>
      
      <div className="space-y-4">
        {recommendations.map((rec, idx) => (
          <div key={idx} className="border rounded-lg p-4 hover:bg-gray-50">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center">
                <span className="font-semibold text-lg">{rec.drug}</span>
                <span className={`ml-3 px-2 py-1 rounded text-sm ${getTierColor(rec.tier)}`}>
                  Tier {rec.tier}
                </span>
              </div>
              <div className="flex items-center">
                <div className="text-right mr-4">
                  <div className="text-sm text-gray-600">ML Confidence</div>
                  <div className="font-semibold">{(rec.confidence * 100).toFixed(0)}%</div>
                </div>
                <div className="text-right mr-4">
                  <div className="text-sm text-gray-600">Response Probability</div>
                  <div className="font-semibold">{(rec.responseProb * 100).toFixed(0)}%</div>
                </div>
                <div className="w-24 bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-blue-600 h-2 rounded-full" 
                    style={{ width: `${rec.confidence * 100}%` }}
                  ></div>
                </div>
              </div>
            </div>
            
            <p className="text-gray-700 mb-3">{rec.rationale}</p>
            
            {rec.agentConsensus && (
              <div className="mb-3 p-3 bg-gray-50 rounded">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-700">Multi-Agent Consensus</span>
                  <button 
                    onClick={() => toggleAgentDetails(idx)}
                    className="text-xs text-blue-600 hover:text-blue-800"
                  >
                    {showAgentDetails[idx] ? 'Hide Details' : 'Show Details'}
                  </button>
                </div>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                  {Object.entries(rec.agentConsensus).map(([agent, consensus]) => (
                    <div key={agent} className={`px-2 py-1 rounded text-xs ${getConsensusColor(consensus)}`}>
                      <div className="font-medium capitalize">{agent.replace('_', ' ')}</div>
                      <div>{consensus}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
            
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <span className="text-sm text-gray-600">
                  Mechanism: {rec.mechanism}
                </span>
                {rec.trials && rec.trials.length > 0 && (
                  <a href="#" className="text-blue-600 text-sm flex items-center hover:underline">
                    <ExternalLink className="w-4 h-4 mr-1" />
                    Active Trial ({rec.trials[0]})
                  </a>
                )}
              </div>
              <button 
                onClick={() => setSelectedRec(selectedRec === idx ? null : idx)}
                className="text-blue-600 hover:text-blue-800 flex items-center"
              >
                {selectedRec === idx ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
                Evidence Details
              </button>
            </div>
            
            {selectedRec === idx && (
              <div className="mt-4 p-4 bg-gray-50 rounded">
                <h4 className="font-semibold mb-3">Detailed Evidence Analysis:</h4>
                {rec.evidence.map((ev, evIdx) => (
                  <div key={evIdx} className="mb-3 p-3 bg-white rounded border-l-4 border-blue-500">
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-medium">{ev.source}</span>
                      {ev.level && <span className="text-sm text-gray-600">Evidence Level: {ev.level}</span>}
                    </div>
                    <p className="text-sm text-gray-700 mb-2">{ev.description}</p>
                    {ev.pmid && (
                      <div className="flex items-center justify-between">
                        <a href={`https://pubmed.ncbi.nlm.nih.gov/${ev.pmid}`} className="text-xs text-blue-600 hover:underline">
                          PMID: {ev.pmid}
                        </a>
                        <span className="text-xs text-gray-500">Validated by Literature Agent</span>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

// Chat Interface Component
const ChatInterface = ({ recommendations }) => {
  const [messages, setMessages] = useState([
    { 
      type: 'system', 
      content: 'Hi! I am the PROMETHEUS AI assistant. I can explain our multi-agent reasoning process and help you understand how we arrived at these recommendations.'
    }
  ]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  
  const handleSend = () => {
    if (!input.trim()) return;
    
    setMessages(prev => [...prev, { type: 'user', content: input }]);
    setIsTyping(true);
    
    setTimeout(() => {
      let response = "I can help explain that aspect of our analysis.";
      
      if (input.toLowerCase().includes('braf')) {
        response = "BRAF V600E Analysis: Our multi-agent system identified this mutation with high confidence. Knowledge Base Consensus: OncoKB Level 1 evidence, CIViC pan-cancer sensitivity. ML Ensemble: 83% response probability. Literature: Found 8 case reports with 75% response rate. Note: Off-guideline for MM but supported by tumor-agnostic evidence.";
      } else if (input.toLowerCase().includes('venetoclax')) {
        response = "Venetoclax Analysis: Recommended based on high BCL2 expression (85th percentile) despite no t(11;14). Expression Agent found BCL2 TPM 45.2 with drug sensitivity score 0.78. Literature shows 48% response in non-t(11;14) cases. ML prediction: 67% response probability.";
      } else if (input.toLowerCase().includes('confidence')) {
        response = "Confidence Methodology: Evidence quality weighting (FDA approval +40%, trials +30%, preclinical +15%, literature +15%). Cross-agent validation requires 2+ agent consensus for Tier 1, 3+ for >80% confidence. Uses Bayesian ensemble averaging.";
      } else if (input.toLowerCase().includes('agent')) {
        response = "Multi-Agent Architecture: 6-stage pipeline with Validation Agents, Knowledge Base Agents, Literature RAG Agents, Guidelines Agents, ML Ensemble Agents, and Consensus Agents. Each agent votes on recommendations, conflicts trigger additional searches.";
      }
      
      setIsTyping(false);
      setMessages(prev => [...prev, { type: 'assistant', content: response }]);
    }, 1500);
    
    setInput('');
  };
  
  return (
    <div className="bg-white rounded-lg border p-6">
      <h2 className="text-xl font-bold mb-4 flex items-center">
        <MessageCircle className="w-5 h-5 mr-2" />
        Ask PROMETHEUS AI
      </h2>
      
      <div className="h-80 overflow-y-auto border rounded p-3 mb-4 space-y-3">
        {messages.map((msg, idx) => (
          <div key={idx} className={msg.type === 'user' ? 'flex justify-end' : 'flex justify-start'}>
            <div className={`max-w-lg p-3 rounded-lg ${
              msg.type === 'user' ? 'bg-blue-600 text-white' : 
              msg.type === 'system' ? 'bg-gray-100 text-gray-700' :
              'bg-gray-200 text-gray-800'
            }`}>
              <div className="text-sm">{msg.content}</div>
            </div>
          </div>
        ))}
        
        {isTyping && (
          <div className="flex justify-start">
            <div className="bg-gray-200 text-gray-800 p-3 rounded-lg">
              <div className="flex items-center space-x-1">
                <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce"></div>
              </div>
            </div>
          </div>
        )}
      </div>
      
      <div className="space-y-3">
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => setInput('How did you analyze the BRAF mutation?')}
            className="px-3 py-1 text-sm bg-blue-50 text-blue-700 rounded-full hover:bg-blue-100"
          >
            BRAF Analysis
          </button>
          <button
            onClick={() => setInput('Why venetoclax without t(11;14)?')}
            className="px-3 py-1 text-sm bg-blue-50 text-blue-700 rounded-full hover:bg-blue-100"
          >
            Venetoclax Logic
          </button>
          <button
            onClick={() => setInput('How confident are these predictions?')}
            className="px-3 py-1 text-sm bg-blue-50 text-blue-700 rounded-full hover:bg-blue-100"
          >
            Confidence Scoring
          </button>
        </div>
        
        <div className="flex space-x-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSend()}
            placeholder="Ask about the multi-agent analysis..."
            className="flex-1 p-2 border rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button 
            onClick={handleSend}
            disabled={isTyping}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
          >
            {isTyping ? 'Processing...' : 'Send'}
          </button>
        </div>
      </div>
    </div>
  );
};

// Export Panel Component
const ExportPanel = ({ patient, recommendations }) => {
  const handleExport = (format) => {
    alert(`Exporting ${format} report for patient ${patient.id}`);
  };
  
  return (
    <div className="bg-white rounded-lg border p-6">
      <h2 className="text-xl font-bold mb-4 flex items-center">
        <Download className="w-5 h-5 mr-2" />
        Export & Sign-off
      </h2>
      
      <div className="space-y-4">
        <div className="flex space-x-3">
          <button 
            onClick={() => handleExport('PDF')}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            Export PDF Report
          </button>
          <button 
            onClick={() => handleExport('FHIR')}
            className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
          >
            Export FHIR Bundle
          </button>
        </div>
        
        <div className="p-4 bg-gray-50 rounded">
          <label className="flex items-center">
            <input type="checkbox" className="mr-2" />
            <span className="text-sm">MTB Review Complete - Ready for clinical decision</span>
          </label>
        </div>
        
        <div className="text-sm text-gray-600">
          <p><strong>Report Summary:</strong></p>
          <p>• {recommendations.length} therapeutic recommendations generated</p>
          <p>• {recommendations.filter(r => r.tier === "1").length} Tier 1 (highest evidence) options</p>
          <p>• {recommendations.filter(r => r.trials && r.trials.length > 0).length} active clinical trials available</p>
        </div>
      </div>
    </div>
  );
};

// Main Application
const PrometheusApp = () => {
  const [recommendations, setRecommendations] = useState([]);
  const [activeTab, setActiveTab] = useState('overview');
  
  const tabs = [
    { id: 'overview', label: 'Overview', icon: FileText },
    { id: 'genomics', label: 'Multi-Omics', icon: Dna },
    { id: 'recommendations', label: 'Recommendations', icon: Star },
    { id: 'chat', label: 'Ask AI', icon: MessageCircle }
  ];
  
  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white border-b px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">PROMETHEUS</h1>
            <p className="text-gray-600">Precision Oncology Decision Support System</p>
          </div>
          <div className="flex items-center space-x-4">
            <div className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm">
              Analysis Complete
            </div>
            <button className="p-2 text-gray-400 hover:text-gray-600">
              <Info className="w-5 h-5" />
            </button>
          </div>
        </div>
      </header>
      
      <nav className="bg-white border-b px-6">
        <div className="flex space-x-8">
          {tabs.map(tab => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`py-4 px-2 border-b-2 font-medium text-sm flex items-center ${
                  activeTab === tab.id 
                    ? 'border-blue-500 text-blue-600' 
                    : 'border-transparent text-gray-500 hover:text-gray-700'
                }`}
              >
                <Icon className="w-4 h-4 mr-2" />
                {tab.label}
              </button>
            );
          })}
        </div>
      </nav>
      
      <main className="p-6">
        {activeTab === 'overview' && (
          <div className="max-w-7xl mx-auto">
            <PatientSnapshot patient={mockPatientData} />
            <ReasoningEngine 
              patient={mockPatientData} 
              onRecommendationsGenerated={setRecommendations}
            />
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <ChatInterface recommendations={recommendations} />
              <ExportPanel patient={mockPatientData} recommendations={recommendations} />
            </div>
          </div>
        )}
        
        {activeTab === 'genomics' && (
          <div className="max-w-7xl mx-auto">
            <MultiOmicViewer 
              variants={mockPatientData.genomicVariants}
              expression={mockPatientData.rnaExpression}
            />
          </div>
        )}
        
        {activeTab === 'recommendations' && recommendations.length > 0 && (
          <div className="max-w-7xl mx-auto">
            <TopRecommendations recommendations={recommendations} />
          </div>
        )}
        
        {activeTab === 'chat' && (
          <div className="max-w-4xl mx-auto">
            <ChatInterface recommendations={recommendations} />
          </div>
        )}
      </main>
    </div>
  );
};

export default PrometheusApp;