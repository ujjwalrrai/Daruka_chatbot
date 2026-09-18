import { useEffect, useMemo, useState } from "react";
import type { FormEvent, ReactNode } from "react";
import {
  Activity,
  ArrowRight,
  BookOpen,
  CheckCircle2,
  Leaf,
  MessageCircle,
  Send,
  Sprout,
  Target,
  Waves,
} from "lucide-react";

const API_URL = "http://localhost:8000";

type SiteState = {
  soc?: string | null;
  rainfall?: number | null;
  land_use?: string | null;
  crop?: string | null;
  biodiversity?: string | null;
  erosion?: string | null;
};

type Evidence = {
  source_id?: string;
  metric?: string;
  relation?: string;
  effect_type?: string;
  effect_size?: number | string | null;
  effect_unit?: string | null;
  evidence_tier?: string;
  confidence?: number;
  quote?: string;
};

type Source = {
  id?: string;
  title?: string;
  authors?: string;
  year?: number;
  doi?: string | null;
  url?: string | null;
  evidence_tier?: string;
};

type EvidenceCard = {
  intervention?: string;
  impacted_metrics?: string[];
  metric_coverage?: number;
  evidence?: Evidence[];
  sources?: Source[];
  why_it_works?: string | null;
  why_it_works_source?:
    | "cee_mechanism"
    | "llm_scientific_explanation"
    | null;
};

type ChatResponse = {
  response: string;
  session_id: string;
  analysis?: {
    site_state?: SiteState;
    detected_deficiencies?: string[];
    recommendations?: any[];
    conversation?: {
      reused_previous_context?: boolean;
    };
  };
  evidence_cards?: EvidenceCard[];
  needs_clarification?: boolean;
  missing_fields?: string[];
  site_state?: SiteState;
};

type Message = {
  role: "user" | "assistant";
  content: string;
};

const METRIC_LABELS: Record<string, string> = {
  soil_organic_carbon: "Soil Organic Carbon",
  water_holding_capacity: "Water-Holding Capacity",
  biodiversity: "Biodiversity",
  associated_biodiversity: "Associated Biodiversity",
  agricultural_production: "Agricultural Production",
  pest_disease_control: "Pest & Disease Control",
  crop_yield: "Crop Yield",
  soil_quality: "Soil Quality",
  soil_organic_matter: "Soil Organic Matter",
};

const INTERVENTION_ACTIONS: Record<string, string> = {
  conservation_agriculture:
    "Adopt conservation-agriculture practices that maintain soil cover and reduce soil disturbance.",
  cover_crops:
    "Introduce suitable cover crops between cash-crop cycles while monitoring water availability.",
  manure_application:
    "Use appropriately managed manure as an organic soil amendment where locally suitable.",
  residue_retention:
    "Retain suitable crop residues on the field rather than removing or burning them.",
  n_fertilization:
    "Manage nitrogen fertilization according to crop and soil requirements rather than treating it as a universal soil-carbon solution.",
  crop_rotation:
    "Rotate crops across seasons to diversify the cropping system.",
  intercropping:
    "Introduce compatible crop combinations to diversify the production system.",
  crop_diversification:
    "Diversify the crop sequence to reduce dependence on a single crop.",
};

function pretty(value?: string | null) {
  if (!value) return "—";

  return value
    .replaceAll("_", " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function metricLabel(metric?: string) {
  if (!metric) return "Environmental metric";
  return METRIC_LABELS[metric] || pretty(metric);
}

function tierLabel(tier?: string) {
  if (!tier) return "Evidence";

  return tier
    .replace("tier1_meta", "Meta-analysis")
    .replace("tier2", "Research evidence")
    .replace("tier3_institutional", "Institutional source")
    .replaceAll("_", " ");
}

function formatEffect(evidence: Evidence) {
  if (
    evidence.effect_size !== null &&
    evidence.effect_size !== undefined
  ) {
    const size = evidence.effect_size;

    if (evidence.effect_type === "pct_change") {
      return `${size}%`;
    }

    if (evidence.effect_unit) {
      return `${size} ${evidence.effect_unit}`;
    }

    return `${size}`;
  }

  if (evidence.relation) {
    return pretty(evidence.relation);
  }

  return "Qualitative evidence";
}

function interventionLabel(name?: string) {
  if (!name) return "Recommended intervention";

  return pretty(name);
}

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [sessionId, setSessionId] = useState<string | null>(
    () => localStorage.getItem("cee_session_id")
  );

  const [siteState, setSiteState] = useState<SiteState>({});
  const [evidenceCards, setEvidenceCards] = useState<EvidenceCard[]>(
    []
  );

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const hasContext = useMemo(
    () => Object.values(siteState).some((value) => value !== null && value !== undefined),
    [siteState]
  );

  useEffect(() => {
    if (sessionId) {
      localStorage.setItem("cee_session_id", sessionId);
    }
  }, [sessionId]);

  async function sendMessage(event?: FormEvent) {
    event?.preventDefault();

    const message = input.trim();

    if (!message || loading) return;

    setMessages((previous) => [
      ...previous,
      {
        role: "user",
        content: message,
      },
    ]);

    setInput("");
    setLoading(true);
    setError("");

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message,
          session_id: sessionId,
        }),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data: ChatResponse = await response.json();

      setSessionId(data.session_id);

      const nextState =
        data.site_state ||
        data.analysis?.site_state ||
        {};

      setSiteState(nextState);

      if (data.evidence_cards?.length) {
        setEvidenceCards(data.evidence_cards);
      }

      setMessages((previous) => [
        ...previous,
        {
          role: "assistant",
          content: data.response || "No response received.",
        },
      ]);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to reach CEE."
      );
    } finally {
      setLoading(false);
    }
  }

  function newSession() {
    localStorage.removeItem("cee_session_id");
    setSessionId(null);
    setMessages([]);
    setSiteState({});
    setEvidenceCards([]);
    setError("");
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <Leaf size={21} />
          </div>

          <div>
            <div className="brand-name">
              Darukaa.Earth
            </div>

            <div className="brand-subtitle">
              Causal Evidence Engine
            </div>
          </div>
        </div>

        <div className="sidebar-section">
          <div className="sidebar-label">
            CURRENT SESSION
          </div>

          <div className="session-card">
            <div className="status-dot" />
            <div>
              <strong>
                Environmental assessment
              </strong>

              <span>
                Persistent context enabled
              </span>
            </div>
          </div>
        </div>

        <div className="sidebar-section">
          <div className="sidebar-label">
            PIPELINE
          </div>

          <div className="pipeline">
            <PipelineStep
              icon={<MessageCircle size={16} />}
              label="Conversation"
              active
            />

            <PipelineLine />

            <PipelineStep
              icon={<Target size={16} />}
              label="Site context"
              active={hasContext}
            />

            <PipelineLine />

            <PipelineStep
              icon={<Activity size={16} />}
              label="Causal reasoning"
              active={evidenceCards.length > 0}
            />

            <PipelineLine />

            <PipelineStep
              icon={<BookOpen size={16} />}
              label="Scientific evidence"
              active={evidenceCards.length > 0}
            />
          </div>
        </div>

        <button
          className="new-session-button"
          onClick={newSession}
        >
          New assessment
        </button>
      </aside>

      <main className="main">
        <header className="topbar">
          <div>
            <div className="eyebrow">
              AI ENVIRONMENTAL SCIENTIST
            </div>

            <h1>
              Field assessment
            </h1>
          </div>

          <div className="connection">
            <span className="connection-dot" />
            CEE online
          </div>
        </header>

        <div className="content">
          <section className="chat-column">
            <div className="conversation">
              {messages.length === 0 && (
                <Welcome />
              )}

              {messages.map((message, index) => (
                <div
                  key={`${message.role}-${index}`}
                  className={`message-row ${message.role}`}
                >
                  {message.role === "assistant" && (
                    <div className="avatar scientist">
                      <Leaf size={17} />
                    </div>
                  )}

                  <div
                    className={`message ${
                      message.role === "assistant"
                        ? "assistant-message"
                        : "user-message"
                    }`}
                  >
                    {message.content}
                  </div>

                  {message.role === "user" && (
                    <div className="avatar user">
                      U
                    </div>
                  )}
                </div>
              ))}

              {loading && (
                <div className="message-row assistant">
                  <div className="avatar scientist">
                    <Leaf size={17} />
                  </div>

                  <div className="message assistant-message typing">
                    <span />
                    <span />
                    <span />
                  </div>
                </div>
              )}
            </div>

            {error && (
              <div className="error-box">
                {error}
              </div>
            )}

            <form
              className="composer"
              onSubmit={sendMessage}
            >
              <input
                value={input}
                onChange={(event) =>
                  setInput(event.target.value)
                }
                placeholder="Describe your field, soil, rainfall or cropping system..."
              />

              <button
                type="submit"
                disabled={loading || !input.trim()}
                aria-label="Send"
              >
                <Send size={18} />
              </button>
            </form>
          </section>

          <aside className="insights">
            <SiteContext state={siteState} />

            {evidenceCards.length > 0 && (
              <RecommendationPanel
                cards={evidenceCards}
                siteState={siteState}
              />
            )}

            {evidenceCards.length === 0 && (
              <div className="empty-panel">
                <Sprout size={25} />

                <strong>
                  Evidence will appear here
                </strong>

                <span>
                  Once enough field context is available,
                  CEE will connect your conditions to
                  evidence-backed interventions.
                </span>
              </div>
            )}
          </aside>
        </div>
      </main>
    </div>
  );
}

function Welcome() {
  return (
    <div className="welcome">
      <div className="welcome-icon">
        <Leaf size={25} />
      </div>

      <div className="eyebrow">
        DARUKAA.EARTH
      </div>

      <h2>
        Let's understand your field.
      </h2>

      <p>
        Tell me what you are observing — soil carbon,
        rainfall, crops, biodiversity, water retention,
        erosion or your current farming system.
      </p>

      <div className="example-prompts">
        <div>
          <ArrowRight size={14} />
          My soil carbon is low
        </div>

        <div>
          <ArrowRight size={14} />
          I have around 500 mm rainfall
        </div>

        <div>
          <ArrowRight size={14} />
          I grow wheat continuously
        </div>
      </div>
    </div>
  );
}

function SiteContext({
  state,
}: {
  state: SiteState;
}) {
  const fields = [
    {
      key: "soc",
      label: "Soil carbon",
      value: state.soc
        ? pretty(state.soc)
        : null,
      icon: <Sprout size={15} />,
    },
    {
      key: "rainfall",
      label: "Rainfall",
      value:
        state.rainfall !== null &&
        state.rainfall !== undefined
          ? `${state.rainfall} mm`
          : null,
      icon: <Waves size={15} />,
    },
    {
      key: "land_use",
      label: "Land use",
      value: state.land_use
        ? pretty(state.land_use)
        : null,
      icon: <Activity size={15} />,
    },
    {
      key: "crop",
      label: "Crop",
      value: state.crop
        ? pretty(state.crop)
        : null,
      icon: <Leaf size={15} />,
    },
  ];

  return (
    <section className="panel">
      <div className="panel-heading">
        <div>
          <div className="panel-kicker">
            SITE PROFILE
          </div>

          <h3>
            Current field context
          </h3>
        </div>

        <CheckCircle2
          size={18}
          className={
            fields.some((field) => field.value)
              ? "green-icon"
              : "muted-icon"
          }
        />
      </div>

      <div className="context-grid">
        {fields.map((field) => (
          <div
            className={`context-item ${
              field.value ? "filled" : ""
            }`}
            key={field.key}
          >
            <div className="context-icon">
              {field.icon}
            </div>

            <div>
              <span>
                {field.label}
              </span>

              <strong>
                {field.value || "Not provided"}
              </strong>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function RecommendationPanel({
  cards,
  siteState,
}: {
  cards: EvidenceCard[];
  siteState: SiteState;
}) {
  const primary = cards[0];

  if (!primary) return null;

  const intervention =
    primary.intervention || "intervention";

  const action =
    INTERVENTION_ACTIONS[intervention] ||
    `Apply ${interventionLabel(intervention).toLowerCase()} as an evidence-supported management intervention for the identified field conditions.`;

  return (
    <div className="recommendations">
      <div className="recommendation-header">
        <div>
          <div className="panel-kicker">
            CEE RECOMMENDATION
          </div>

          <h2>
            {interventionLabel(intervention)}
          </h2>
        </div>

        <div className="recommendation-badge">
          Evidence-backed
        </div>
      </div>

      <div className="assessment-strip">
        <Target size={17} />

        <div>
          <strong>
            Context-aware assessment
          </strong>

          <span>
            CEE used the accumulated field context rather
            than treating the query as an isolated question.
          </span>
        </div>
      </div>

      <div className="section-card">
        <SectionTitle
          number="01"
          title="What to do"
        />

        <p className="body-copy">
          {action}
        </p>
      </div>

      <div className="section-card">
        <SectionTitle
          number="02"
          title="Why it works"
        />

        <p className="body-copy">
          {primary.why_it_works ||
            "CEE does not currently contain an explicit causal mechanism for this intervention. The recommendation is based on the retrieved evidence shown below; no additional mechanism has been assumed."}
        </p>

        {primary.why_it_works_source ===
          "llm_scientific_explanation" && (
          <small className="evidence-note">
            General scientific explanation generated by the AI;
            it is not a retrieved citation or study result.
          </small>
        )}

        {primary.why_it_works_source ===
          "cee_mechanism" && (
          <small className="evidence-note">
            Mechanism derived from the CEE causal graph.
          </small>
        )}
      </div>

      <div className="section-card">
        <SectionTitle
          number="03"
          title="Impacted environmental metrics"
        />

        <div className="metric-list">
          {(primary.impacted_metrics || []).map(
            (metric) => {
              const evidence =
                primary.evidence?.find(
                  (item) => item.metric === metric
                );

              return (
                <div
                  className="metric-row"
                  key={metric}
                >
                  <div>
                    <span>
                      {metricLabel(metric)}
                    </span>

                    <small>
                      {evidence?.evidence_tier
                        ? tierLabel(
                            evidence.evidence_tier
                          )
                        : "Evidence connected"}
                    </small>
                  </div>

                  <strong>
                    {evidence
                      ? formatEffect(evidence)
                      : "Increase"}
                  </strong>
                </div>
              );
            }
          )}
        </div>
      </div>

      <div className="section-card">
        <SectionTitle
          number="04"
          title="Scientific evidence"
        />

        <div className="evidence-list">
          {(primary.evidence || []).map(
            (evidence, index) => (
              <EvidenceItem
                key={`${evidence.source_id}-${evidence.metric}-${index}`}
                evidence={evidence}
              />
            )
          )}
        </div>
      </div>

      <div className="section-card">
        <SectionTitle
          number="05"
          title="Scientific references"
        />

        <div className="source-list">
          {(primary.sources || []).map(
            (source) => (
              <SourceItem
                key={source.id}
                source={source}
              />
            )
          )}
        </div>
      </div>

      <div className="context-footer">
        <div className="panel-kicker">
          CONTEXT CONSIDERED
        </div>

        <div className="context-tags">
          {siteState.soc && (
            <span>
              Low soil carbon
            </span>
          )}

          {siteState.rainfall !== null &&
            siteState.rainfall !== undefined && (
              <span>
                {siteState.rainfall} mm rainfall
              </span>
            )}

          {siteState.land_use && (
            <span>
              {pretty(siteState.land_use)}
            </span>
          )}

          {siteState.crop && (
            <span>
              {pretty(siteState.crop)}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

function SectionTitle({
  number,
  title,
}: {
  number: string;
  title: string;
}) {
  return (
    <div className="section-title">
      <span>
        {number}
      </span>

      <h3>
        {title}
      </h3>
    </div>
  );
}

function EvidenceItem({
  evidence,
}: {
  evidence: Evidence;
}) {
  return (
    <div className="evidence-item">
      <div className="evidence-top">
        <div>
          <strong>
            {metricLabel(evidence.metric)}
          </strong>

          <span>
            {evidence.source_id}
          </span>
        </div>

        <div className="effect">
          {formatEffect(evidence)}
        </div>
      </div>

      <div className="evidence-meta">
        {tierLabel(evidence.evidence_tier)}

        {evidence.confidence !== undefined &&
          ` · ${(evidence.confidence * 100).toFixed(0)}% confidence`}
      </div>

      {evidence.quote && (
        <p>
          "{cleanQuote(evidence.quote)}"
        </p>
      )}
    </div>
  );
}

function SourceItem({
  source,
}: {
  source: Source;
}) {
  return (
    <div className="source-item">
      <div className="source-id">
        {source.id}
      </div>

      <div className="source-content">
        <strong>
          {source.title || "Scientific source"}
        </strong>

        <span>
          {source.authors || "Source"} ·{" "}
          {source.year || "n.d."}
        </span>

        <small>
          {tierLabel(source.evidence_tier)}
        </small>
      </div>

      {source.url && (
        <a
          href={source.url}
          target="_blank"
          rel="noreferrer"
        >
          View
        </a>
      )}
    </div>
  );
}

function PipelineStep({
  icon,
  label,
  active,
}: {
  icon: ReactNode;
  label: string;
  active?: boolean;
}) {
  return (
    <div
      className={`pipeline-step ${
        active ? "active" : ""
      }`}
    >
      <div className="pipeline-icon">
        {icon}
      </div>

      <span>
        {label}
      </span>
    </div>
  );
}

function PipelineLine() {
  return <div className="pipeline-line" />;
}

function cleanQuote(quote: string) {
  return quote
    .replaceAll("â€“", "–")
    .replaceAll("â€”", "—")
    .replaceAll("â€™", "’")
    .replaceAll("â€œ", "“")
    .replaceAll("â€", "”")
    .replaceAll("âˆ’", "−")
    .replaceAll("â‰¥", "≥")
    .replaceAll("â‰¤", "≤")
    .replaceAll("â€“", "–");
}

export default App;