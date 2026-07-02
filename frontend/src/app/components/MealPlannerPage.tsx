import { useState, useRef, useEffect, type ReactNode, type ChangeEvent, type MouseEvent } from 'react';
import {
  ChefHat,
  Mic,
  MicOff,
  SendHorizontal,
  Plus,
  User,
  BookOpen,
  X,
  MessageSquare,
  Paperclip,
} from 'lucide-react';

import { authFetch, authHeaders, getAuthToken } from '../../utils/auth';
import { EGFR_RANGES, formatLimit, getLimits } from '../../utils/clinicalConstants';
import { formatStageDisplay } from '../../utils/riskDisplay';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const TEAL = '#2E86AB';

const SAMPLE_PROMPTS = [
  'Is cassava safe for me?',
  'Give me dinner ideas',
  'What foods are high in potassium?',
  'What are safe snacks for me?',
] as const;

const toSourceLabel = (filename: string): string => {
  const map: Record<string, string> = {
    'food database safe G3b.txt': 'Kenya Food Composition Tables 2018',
    'food database safe G3a.txt': 'Kenya Food Composition Tables 2018',
    'food database safe G4.txt': 'Kenya Food Composition Tables 2018',
    'food database safe G2.txt': 'Kenya Food Composition Tables 2018',
    'kidney_diet_guidelines.txt': 'KDOQI Clinical Guidelines 2020',
    kdoqi: 'KDOQI Clinical Guidelines 2020',
    nhanes: 'NHANES Dietary Reference Data',
  };

  const lower = filename.toLowerCase().replace(/_/g, ' ');
  for (const [key, label] of Object.entries(map)) {
    if (lower.includes(key.toLowerCase())) {
      return label;
    }
  }

  return filename
    .replace(/_/g, ' ')
    .replace('.txt', '')
    .replace('.pdf', '')
    .trim();
};

const STAGE_SAFE_RANGES: Record<string, string> = {
  G2: '1-2',
  G3a: '1-3',
  G3b: '1-3',
  G4: '1-4',
};

interface MealPlannerProps {
  isDark: boolean;
  theme: Record<string, string>;
}

interface PatientProfile {
  name: string;
  ckd_stage: string;
  weight_kg: number;
  user_id: string;
}

interface MealItem {
  food: string;
  amount: string;
  k: string;
  p: string;
  pro: string;
  na?: string;
  category?: string;
}

interface UserMessage {
  role: 'user';
  text: string;
}

interface AIMessage {
  role: 'ai';
  text: string;
  meal?: MealItem[];
  sources?: string[];
  occasion?: string;
}

type ChatMessage = UserMessage | AIMessage;

interface ChatSession {
  id: string;
  title: string;
  preview: string;
  messages: ChatMessage[];
  created_at: string;
  updated_at: string;
}

interface MealCardProps {
  items: MealItem[];
  isDark: boolean;
  theme: Record<string, string>;
  token: string;
  occasion?: string;
  ckdStage?: string;
}

function detectOccasion(text: string): string {
  const lower = text.toLowerCase();
  if (lower.includes('breakfast')) return 'Breakfast';
  if (lower.includes('dinner')) return 'Dinner';
  if (lower.includes('snack')) return 'Snack';
  if (lower.includes('lunch')) return 'Lunch';
  return 'Lunch';
}

function stripMarkdown(line: string): string {
  return line.replace(/\*\*/g, '').trim();
}

function applyInlineMarkdown(line: string): string {
  return line
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>');
}

function renderMarkdown(text: string, theme: Record<string, string>) {
  const DAY_NAMES = [
    'Monday',
    'Tuesday',
    'Wednesday',
    'Thursday',
    'Friday',
    'Saturday',
    'Sunday',
  ];

  const lines = text.split('\n');
  const elements: ReactNode[] = [];
  let bulletBuffer: ReactNode[] = [];

  const flushBullets = () => {
    if (bulletBuffer.length > 0) {
      elements.push(
        <ul
          key={`ul-${elements.length}`}
          style={{ listStyle: 'none', padding: 0, margin: '4px 0' }}
        >
          {bulletBuffer}
        </ul>,
      );
      bulletBuffer = [];
    }
  };

  lines.forEach((rawLine, i) => {
    const trimmed = rawLine.trim();
    const plain = stripMarkdown(trimmed);

    if (!trimmed) {
      flushBullets();
      elements.push(<br key={i} />);
      return;
    }

    if (DAY_NAMES.some((d) => plain.startsWith(d))) {
      flushBullets();
      elements.push(
        <p
          key={i}
          className="font-medium mt-4 mb-2"
          style={{ fontSize: '0.875rem', color: theme.text }}
        >
          {plain}
        </p>,
      );
      return;
    }

    if (/^\*?(Breakfast|Lunch|Dinner|Snack)\*?:/i.test(trimmed)) {
      flushBullets();
      elements.push(
        <p
          key={i}
          className="font-medium mt-2 mb-1"
          style={{ fontSize: '0.8125rem', color: TEAL }}
        >
          {plain}
        </p>,
      );
      return;
    }

    if (trimmed.startsWith('* ')) {
      bulletBuffer.push(
        <li
          key={i}
          className="ml-4"
          style={{
            fontSize: '0.875rem',
            lineHeight: 1.75,
            color: theme.text,
            listStyleType: 'disc',
            marginBottom: 2,
          }}
        >
          <span
            dangerouslySetInnerHTML={{
              __html: applyInlineMarkdown(trimmed.slice(2)),
            }}
          />
        </li>,
      );
      return;
    }

    if (trimmed.startsWith('+ ')) {
      bulletBuffer.push(
        <li
          key={i}
          className="ml-8"
          style={{
            fontSize: '0.8125rem',
            lineHeight: 1.75,
            color: theme.textSecondary,
            listStyleType: 'circle',
            marginBottom: 2,
          }}
          dangerouslySetInnerHTML={{
            __html: applyInlineMarkdown(trimmed.slice(2)),
          }}
        />,
      );
      return;
    }

    if (trimmed.startsWith('- ')) {
      bulletBuffer.push(
        <li
          key={i}
          className="ml-4"
          style={{
            fontSize: '0.875rem',
            lineHeight: 1.75,
            color: theme.text,
            listStyleType: 'disc',
            marginBottom: 2,
          }}
          dangerouslySetInnerHTML={{
            __html: applyInlineMarkdown(trimmed.slice(2)),
          }}
        />,
      );
      return;
    }

    flushBullets();
    elements.push(
      <p
        key={i}
        className="mb-2"
        style={{ fontSize: '0.875rem', lineHeight: 1.75, color: theme.text }}
        dangerouslySetInnerHTML={{ __html: applyInlineMarkdown(rawLine) }}
      />,
    );
  });

  flushBullets();
  return elements;
}

function SourceBadge({ source }: { source: string }) {
  return (
    <span
      className="text-[11px] px-2 py-0.5 rounded-full"
      style={{
        border: `1px solid ${TEAL}`,
        color: TEAL,
        background: 'rgba(46,134,171,0.06)',
      }}
    >
      📚 {toSourceLabel(source)}
    </span>
  );
}

function MealCard({
  items,
  isDark,
  theme,
  token,
  occasion = 'Lunch',
  ckdStage = 'G3b',
}: MealCardProps) {
  const [added, setAdded] = useState<Record<number, boolean>>({});
  const [allAdded, setAllAdded] = useState(false);

  const stageRange = STAGE_SAFE_RANGES[ckdStage] ?? '1-3';

  const addFoodToLog = async (item: MealItem, index: number) => {
    try {
      const sodiumMg = item.na
        ? parseFloat(item.na.replace('mg', ''))
        : 0;

      const res = await authFetch(`${API_BASE}/api/patient/food-log`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          food_name: item.food.replace(/^[⚠\s]+/, '').split('(')[0].trim(),
          portion_grams: parseFloat(item.amount.replace('g', '')),
          meal_occasion: occasion,
          potassium_mg: parseFloat(item.k.replace('mg', '')),
          phosphorus_mg: parseFloat(item.p.replace('mg', '')),
          protein_g: parseFloat(item.pro.replace('g', '')),
          sodium_mg: sodiumMg,
          category: item.category || 'Other',
          stage_safe_range: stageRange,
        }),
      });
      if (!res.ok) {
        throw new Error(`Food log request failed (${res.status})`);
      }
      setAdded((a) => ({ ...a, [index]: true }));
    } catch (err) {
      console.error('Add food failed:', err);
    }
  };

  const addAll = async () => {
    await Promise.all(items.map((item, i) => addFoodToLog(item, i)));
    setAllAdded(true);
  };

  return (
    <div className="mt-3 w-full overflow-x-auto">
      <div
        className="rounded-lg overflow-hidden text-xs min-w-[280px]"
        style={{ border: `1px solid ${theme.cardBorder}` }}
      >
        <div
          className="grid gap-2 px-3 py-2 font-semibold"
          style={{
            gridTemplateColumns: '1.4fr 0.7fr 0.6fr 0.6fr 0.5fr',
            background: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.04)',
            color: theme.textSecondary,
            fontSize: '0.65rem',
          }}
        >
          <span>Food</span>
          <span>Amount</span>
          <span>Potassium (mg)</span>
          <span>Phosphorus (mg)</span>
          <span>Protein (g)</span>
        </div>
        {items.map((item, i) => (
          <div
            key={`${item.food}-${i}`}
            className="grid gap-2 px-3 py-2 items-center"
            style={{
              gridTemplateColumns: '1.4fr 0.7fr 0.6fr 0.6fr 0.5fr',
              borderTop: `1px solid ${theme.cardBorder}`,
              background: i % 2 === 0 ? 'transparent' : isDark ? 'rgba(255,255,255,0.02)' : 'rgba(0,0,0,0.02)',
            }}
          >
            <span className="font-medium capitalize" style={{ color: theme.text }}>
              {item.food}
            </span>
            <span style={{ color: theme.textSecondary }}>{item.amount}</span>
            <span style={{ color: TEAL, fontWeight: 600 }}>{item.k}</span>
            <span style={{ color: '#F39C12', fontWeight: 600 }}>{item.p}</span>
            <span style={{ color: '#27AE60', fontWeight: 600 }}>{item.pro}</span>
          </div>
        ))}
      </div>

      <button
        type="button"
        onClick={() => void addAll()}
        disabled={allAdded}
        className="w-full mt-3 py-2.5 rounded-xl text-white text-sm font-semibold flex items-center justify-center gap-2 transition-opacity disabled:opacity-60"
        style={{ background: 'linear-gradient(135deg, #2E86AB 0%, #1A5F7A 100%)' }}
      >
        <Plus size={15} />
        {allAdded ? '✓ All added to log' : "Add all to today's log"}
      </button>

      <div className="flex flex-wrap gap-2 mt-2 items-center">
        {items.slice(0, 4).map((item, i) => (
          <button
            key={`add-${item.food}-${i}`}
            type="button"
            onClick={() => void addFoodToLog(item, i)}
            disabled={added[i] || allAdded}
            className="text-xs px-2.5 py-1.5 rounded-full transition-colors disabled:opacity-70"
            style={{
              border: `1px solid ${added[i] || allAdded ? '#27AE60' : theme.cardBorder}`,
              color: added[i] || allAdded ? '#27AE60' : TEAL,
              background: added[i] || allAdded ? 'rgba(39,174,96,0.1)' : 'transparent',
            }}
          >
            {added[i] || allAdded ? '✓ Added' : `+ Add ${item.food} ${item.amount}`}
          </button>
        ))}
        {items.length > 4 && (
          <span style={{ color: theme.textTertiary, fontSize: '0.72rem' }}>
            +{items.length - 4} more included in &quot;Add all&quot;
          </span>
        )}
      </div>
    </div>
  );
}

export function MealPlanner({ isDark, theme }: MealPlannerProps) {
  const [profile, setProfile] = useState<PatientProfile | null>(null);
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isTyping, setIsTyping] = useState(false);
  const [loadingStage, setLoadingStage] = useState<'thinking' | 'searching' | 'slow' | null>(null);
  const [uploadedText, setUploadedText] = useState<string | null>(null);
  const [isListening, setIsListening] = useState(false);
  const [voiceSupported, setVoiceSupported] = useState(true);
  const [chatHistory, setChatHistory] = useState<ChatSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const recognitionRef = useRef<any>(null);
  const currentSessionIdRef = useRef<string | null>(null);

  const token = getAuthToken() ?? '';

  const jsonAuthHeaders = authHeaders({ 'Content-Type': 'application/json' });

  const refreshChatHistory = async () => {
    if (!token) return;
    try {
      const res = await authFetch(`${API_BASE}/api/meal-planner/sessions`, {
        headers: jsonAuthHeaders,
      });
      if (res.status === 401) return;
      if (res.ok) {
        setChatHistory(await res.json());
      }
    } catch (err) {
      console.error('Load chat history failed:', err);
    }
  };

  useEffect(() => {
    if (!token) return;
    authFetch(`${API_BASE}/api/patient/profile`)
      .then((r) => {
        if (r.status === 401) return null;
        return r.json();
      })
      .then((data) => {
        if (!data) return;
        setProfile({
          name: data.name || 'Patient',
          ckd_stage: data.ckd_stage || 'G3b',
          weight_kg: data.weight_kg || 65,
          user_id: localStorage.getItem('guidaplate_user_id') || '',
        });
      })
      .catch(console.error);
  }, [token]);

  useEffect(() => {
    if (!token) return;
    void refreshChatHistory();
  }, [token]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  useEffect(() => {
    if (!isTyping) {
      setLoadingStage(null);
      return;
    }

    setLoadingStage('thinking');

    const t1 = setTimeout(() => setLoadingStage('searching'), 2000);
    const t2 = setTimeout(() => setLoadingStage('slow'), 6000);

    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
    };
  }, [isTyping]);

  useEffect(() => {
    currentSessionIdRef.current = currentSessionId;
  }, [currentSessionId]);

  const stage = profile?.ckd_stage || 'G3b';
  const lims = getLimits(stage);
  const proteinLimit = Math.round((profile?.weight_kg || 65) * lims.protein);
  const gfrRange = EGFR_RANGES[stage as keyof typeof EGFR_RANGES] ?? EGFR_RANGES.G3b;

  const limits = [
    { label: 'Potassium limit', value: formatLimit(lims.potassium), unit: 'mg/day', color: TEAL },
    { label: 'Phosphorus limit', value: formatLimit(lims.phosphorus), unit: 'mg/day', color: '#F39C12' },
    { label: 'Protein limit', value: proteinLimit.toString(), unit: 'g/day', color: '#27AE60' },
    { label: 'Sodium limit', value: formatLimit(lims.sodium), unit: 'mg/day', color: '#E74C3C' },
  ];

  const patientLabel = `${profile?.name || 'Patient'} · #${(profile?.user_id || '').slice(0, 6).toUpperCase()}`;

  const initSpeech = () => {
    const SpeechRecognition =
      (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setVoiceSupported(false);
      return;
    }
    const recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';
    recognition.onresult = (event: any) => {
      const transcript = String(event.results?.[0]?.[0]?.transcript ?? '').trim();
      if (transcript) setInput((prev) => (prev ? `${prev} ${transcript}` : transcript));
      setIsListening(false);
    };
    recognition.onerror = () => setIsListening(false);
    recognition.onend = () => setIsListening(false);
    recognitionRef.current = recognition;
  };

  const toggleVoice = () => {
    if (!recognitionRef.current) initSpeech();
    if (!recognitionRef.current) return;
    if (isListening) {
      recognitionRef.current.stop();
      setIsListening(false);
    } else {
      recognitionRef.current.start();
      setIsListening(true);
    }
  };

  const saveToHistory = async (messagesToSave?: ChatMessage[]) => {
    const msgs = messagesToSave ?? messages;
    if (msgs.length === 0 || !token) return;

    const firstUserMsg = msgs.find((m): m is UserMessage => m.role === 'user');
    const lastMsg = msgs[msgs.length - 1];

    const sessionData = {
      session_id: currentSessionIdRef.current || undefined,
      title: firstUserMsg?.text.slice(0, 60) || 'Conversation',
      preview: `${lastMsg.text.slice(0, 80)}...`,
      messages: msgs.map((m) => {
        if (m.role === 'user') {
          return { role: m.role, text: m.text };
        }
        return {
          role: m.role,
          text: m.text,
          meal: m.meal || null,
          sources: m.sources || [],
          occasion: m.occasion || null,
        };
      }),
    };

    try {
      const res = await authFetch(`${API_BASE}/api/meal-planner/sessions`, {
        method: 'POST',
        headers: jsonAuthHeaders,
        body: JSON.stringify(sessionData),
      });
      if (res.status === 401) return;
      if (!res.ok) {
        throw new Error(`Save failed (${res.status})`);
      }
      const data = await res.json();
      setCurrentSessionId(data.id);
      currentSessionIdRef.current = data.id;
      await refreshChatHistory();
    } catch (err) {
      console.error('Save session failed:', err);
    }
  };

  const startNewConversation = async () => {
    await saveToHistory();
    setMessages([]);
    setCurrentSessionId(null);
    currentSessionIdRef.current = null;
    setUploadedText(null);
    setInput('');
  };

  const loadChat = (chat: ChatSession) => {
    setMessages(chat.messages);
    setCurrentSessionId(chat.id);
    currentSessionIdRef.current = chat.id;
    setInput('');
  };

  const deleteSession = async (sessionId: string, e: MouseEvent) => {
    e.stopPropagation();
    if (!token) return;

    try {
      const res = await authFetch(`${API_BASE}/api/meal-planner/sessions/${sessionId}`, {
        method: 'DELETE',
      });
      if (res.status === 401) return;
      if (!res.ok) {
        throw new Error(`Delete failed (${res.status})`);
      }
      setChatHistory((prev) => prev.filter((c) => c.id !== sessionId));
      if (currentSessionIdRef.current === sessionId) {
        setMessages([]);
        setCurrentSessionId(null);
        currentSessionIdRef.current = null;
      }
    } catch (err) {
      console.error('Delete session failed:', err);
    }
  };

  const handleFileUpload = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (file.type.startsWith('image/')) {
      setInput("I've uploaded a photo of my meal. Please analyze it for kidney-safe choices.");
    } else {
      const reader = new FileReader();
      reader.onload = (ev) => {
        const text = ev.target?.result as string;
        setInput(
          `Here is my meal list:\n${text.slice(0, 500)}\nPlease suggest kidney-safe options.`,
        );
      };
      reader.readAsText(file);
    }
    e.target.value = '';
  };

  const sendMessage = async (text: string) => {
    if (!text.trim() || isTyping) return;

    const userMsg: UserMessage = { role: 'user', text: text.trim() };
    setMessages((m) => [...m, userMsg]);
    setInput('');
    setIsTyping(true);

    try {
      const res = await authFetch(`${API_BASE}/api/meal-planner/chat`, {
        method: 'POST',
        headers: jsonAuthHeaders,
        body: JSON.stringify({
          message: text.trim(),
          ckd_stage: profile?.ckd_stage || 'G3b',
          weight_kg: profile?.weight_kg || 65,
          uploaded_text: uploadedText,
          conversation_history: messages.slice(-6).map((m) => ({
            role: m.role === 'user' ? 'user' : 'assistant',
            content: m.text,
          })),
        }),
      });

      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}));
        const detail = errBody.detail;
        const message =
          typeof detail === 'string'
            ? detail
            : Array.isArray(detail)
              ? detail.map((d: { msg?: string }) => d.msg).filter(Boolean).join(', ')
              : `Request failed (${res.status})`;
        throw new Error(message || `Request failed (${res.status})`);
      }

      const data = await res.json();

      const foods = data.suggested_foods || [];

      const seen = new Set<string>();
      const mealItems: MealItem[] = foods
        .filter((f: { english?: string }) => {
          if (!f.english || seen.has(f.english)) return false;
          seen.add(f.english);
          return true;
        })
        .map((f: {
          english: string;
          portion_grams: number;
          potassium_mg: number;
          phosphorus_mg: number;
          protein_g: number;
          sodium_mg?: number;
          category?: string;
        }) => ({
          food: f.english,
          amount: `${f.portion_grams}g`,
          k: `${Math.round(f.potassium_mg)}mg`,
          p: `${Math.round(f.phosphorus_mg)}mg`,
          pro: `${Number(f.protein_g).toFixed(1)}g`,
          na: f.sodium_mg != null ? `${Math.round(f.sodium_mg)}mg` : '0mg',
          category: f.category || 'Other',
        }))
        .slice(0, 6);

      const displayText =
        data.answer ||
        'Here are my recommendations based on your kidney condition and dietary guidelines.';

      const aiResp: AIMessage = {
        role: 'ai',
        text: displayText,
        meal: mealItems.length > 0 ? mealItems : undefined,
        occasion: mealItems.length > 0 ? detectOccasion(text.trim()) : undefined,
        sources: (data.sources || [])
          .map((s: { source?: string }) => s.source)
          .filter(Boolean)
          .slice(0, 2),
      };

      setMessages((m) => {
        const next = [...m, aiResp];
        void saveToHistory(next);
        return next;
      });
    } catch (err) {
      console.error(err);
      const errText =
        err instanceof Error && err.message
          ? err.message
          : 'Sorry, I could not reach the meal planner service. Please check your connection and try again.';
      setMessages((m) => {
        const next = [
          ...m,
          {
            role: 'ai' as const,
            text: errText,
          },
        ];
        void saveToHistory(next);
        return next;
      });
    } finally {
      setIsTyping(false);
    }
  };

  const sidebarBg = isDark ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0.02)';
  const borderColor = theme.cardBorder;

  return (
    <div className="h-full flex flex-col overflow-hidden">
      <div className="flex flex-1 overflow-hidden flex-row min-h-0">
      {/* Left sidebar */}
      <aside
        className="h-full overflow-y-auto flex-shrink-0 w-80 flex flex-col gap-3 p-4"
        style={{
          minWidth: '320px',
          width: '320px',
          borderRight: `1px solid ${borderColor}`,
          background: sidebarBg,
        }}
      >
        {/* Card 1 — Patient profile */}
        <div
          className="rounded-xl p-3 space-y-3"
          style={{ background: theme.cardBg, border: `1px solid ${borderColor}` }}
        >
          <div className="flex items-start justify-between gap-2">
            <div className="flex items-start gap-2 min-w-0">
              <User size={16} style={{ color: TEAL, flexShrink: 0, marginTop: 2 }} />
              <div className="min-w-0">
                <p className="font-semibold text-sm" style={{ color: theme.text }}>
                  Patient Profile
                </p>
                <p className="text-[11px] truncate" style={{ color: theme.textSecondary }}>
                  {patientLabel}
                </p>
              </div>
            </div>
            <span
              className="shrink-0 text-[11px] font-semibold px-2 py-0.5 rounded-full"
              style={{ background: 'rgba(46,134,171,0.12)', color: TEAL }}
            >
              {formatStageDisplay(stage)}
            </span>
          </div>

          <div
            className="flex justify-between items-center px-3 py-2 rounded-lg text-xs"
            style={{
              background: isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.04)',
            }}
          >
            <span style={{ color: theme.textSecondary }}>Kidney Function Score</span>
            <span className="font-semibold" style={{ color: TEAL }}>
              {gfrRange} mL/min/1.73m²
            </span>
          </div>

          <div className="space-y-1.5">
            {limits.map((lim) => (
              <div key={lim.label} className="flex justify-between items-center text-xs">
                <span style={{ color: theme.textSecondary }}>{lim.label}</span>
                <span className="font-semibold" style={{ color: lim.color }}>
                  {lim.value} {lim.unit}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Card 2 — Chat history */}
        <div
          className="rounded-xl p-3 flex-1 flex flex-col min-h-0 overflow-hidden"
          style={{ background: theme.cardBg, border: `1px solid ${borderColor}` }}
        >
          <div className="flex flex-col flex-1 overflow-hidden">
            <p
              className="text-xs font-semibold uppercase tracking-wider mb-2"
              style={{ color: theme.textSecondary }}
            >
              Recent chats
            </p>

            <button
              type="button"
              onClick={() => void startNewConversation()}
              className="flex items-center gap-2 w-full px-3 py-2 rounded-lg border text-sm text-muted-foreground hover:bg-muted/50 mb-3 transition-colors"
              style={{ borderColor }}
            >
              <Plus className="w-4 h-4" />
              New conversation
            </button>

            <div className="flex flex-col gap-1 overflow-y-auto flex-1">
              {chatHistory.length === 0 ? (
                <p
                  className="text-xs text-center mt-4"
                  style={{ color: theme.textSecondary }}
                >
                  No previous conversations
                </p>
              ) : (
                chatHistory.map((chat) => (
                  <div
                    key={chat.id}
                    className="flex items-start gap-2 px-3 py-2 rounded-lg hover:bg-muted/50 transition-colors group"
                  >
                    <button
                      type="button"
                      onClick={() => loadChat(chat)}
                      className="flex items-start gap-2 flex-1 min-w-0 text-left"
                    >
                      <MessageSquare
                        className="w-3.5 h-3.5 mt-0.5 flex-shrink-0"
                        style={{ color: theme.textSecondary }}
                      />
                      <div className="flex-1 min-w-0">
                        <p
                          className="text-xs truncate font-medium"
                          style={{ color: theme.text }}
                        >
                          {chat.title}
                        </p>
                        <p
                          className="text-xs truncate"
                          style={{ color: theme.textSecondary }}
                        >
                          {chat.preview}
                        </p>
                      </div>
                    </button>
                    <button
                      type="button"
                      onClick={(e) => void deleteSession(chat.id, e)}
                      className="opacity-0 group-hover:opacity-100 p-1 rounded hover:text-red-500 transition-opacity flex-shrink-0"
                      style={{ color: theme.textSecondary }}
                      aria-label="Delete conversation"
                    >
                      <X className="w-3 h-3" />
                    </button>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>

        {/* Bottom bar */}
        <div className="flex items-start gap-2 mt-auto pt-1">
          <BookOpen size={13} style={{ color: TEAL, flexShrink: 0, marginTop: 1 }} />
          <p className="text-[11px] leading-relaxed" style={{ color: theme.textSecondary }}>
            Based on international kidney health guidelines
          </p>
        </div>
      </aside>

      {/* Right chat area */}
      <main className="flex-1 flex flex-col h-full overflow-hidden min-h-0 min-w-0">
        {/* Header */}
        <div
          className="flex items-center justify-between px-6 py-3 flex-shrink-0"
          style={{ borderBottom: `1px solid ${borderColor}` }}
        >
          <div className="flex items-center gap-3">
            <div
              className="w-9 h-9 rounded-full flex items-center justify-center shrink-0"
              style={{ background: 'linear-gradient(135deg, #2E86AB 0%, #1A5F7A 100%)' }}
            >
              <ChefHat size={16} className="text-white" />
            </div>
            <div>
              <p className="font-semibold text-sm" style={{ color: theme.text }}>
                Meal Suggestions
              </p>
              <p className="text-xs flex items-center gap-1.5" style={{ color: theme.textSecondary }}>
                <span className="w-1.5 h-1.5 rounded-full bg-green-500" />
                Online · Kidney Diet Assistant
              </p>
            </div>
          </div>
          {messages.length > 0 && (
            <button
              type="button"
              onClick={() => {
                setMessages([]);
                setUploadedText(null);
              }}
              className="text-xs px-3 py-1 rounded-full flex items-center gap-1 transition-opacity hover:opacity-70"
              style={{ border: `1px solid ${borderColor}`, color: theme.textSecondary }}
            >
              <X size={12} /> Clear
            </button>
          )}
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto min-h-0 px-6 py-6 flex flex-col gap-6">
          {messages.length === 0 && !isTyping && (
            <div className="flex-1 flex flex-col items-center justify-center text-center px-4 min-h-0">
              <div
                className="w-16 h-16 rounded-2xl flex items-center justify-center mb-5"
                style={{ background: 'rgba(46,134,171,0.12)' }}
              >
                <ChefHat size={48} style={{ color: TEAL }} />
              </div>
              <p className="font-semibold text-lg mb-2" style={{ color: theme.text }}>
                Ask me about safe meals for your kidneys
              </p>
              <p className="text-sm max-w-md leading-relaxed mb-4" style={{ color: theme.textSecondary }}>
                Ask about safe foods, meal plans, or nutrient limits for your kidney condition.
              </p>
              <span
                className="text-xs px-3 py-1 rounded-full"
                style={{
                  border: `1px solid ${TEAL}`,
                  color: TEAL,
                  background: 'rgba(46,134,171,0.06)',
                }}
              >
                Based on international kidney health guidelines
              </span>
            </div>
          )}

          {messages.map((msg, idx) =>
            msg.role === 'user' ? (
              <div key={idx} className="flex justify-end">
                <div
                  className="max-w-[70%] ml-auto px-4 py-2.5 rounded-full text-sm leading-relaxed text-white"
                  style={{ background: 'linear-gradient(135deg, #2E86AB 0%, #1A5F7A 100%)' }}
                >
                  {msg.text}
                </div>
              </div>
            ) : (
              <div key={idx} className="flex gap-3 items-start">
                <div
                  className="w-7 h-7 rounded-full flex items-center justify-center shrink-0 mt-0.5"
                  style={{ background: 'linear-gradient(135deg, #2E86AB 0%, #1A5F7A 100%)' }}
                >
                  <ChefHat size={14} className="text-white" />
                </div>
                <div className="flex-1 min-w-0">
                  <div>{renderMarkdown(msg.text, theme)}</div>
                  {msg.meal && msg.meal.length > 0 && token && (
                    <MealCard
                      items={msg.meal}
                      isDark={isDark}
                      theme={theme}
                      token={token}
                      occasion={msg.occasion ?? 'Lunch'}
                      ckdStage={stage}
                    />
                  )}
                  {msg.sources && msg.sources.length > 0 && (
                    <div className="mt-3 flex flex-wrap items-center gap-2">
                      <span className="text-[11px]" style={{ color: theme.textSecondary }}>
                        Sources:
                      </span>
                      {msg.sources.map((src) => (
                        <SourceBadge key={src} source={src} />
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ),
          )}

          {isTyping && (
            <div className="flex gap-3 items-start">
              <div
                className="w-7 h-7 rounded-full flex items-center justify-center shrink-0"
                style={{ background: 'linear-gradient(135deg, #2E86AB 0%, #1A5F7A 100%)' }}
              >
                <ChefHat size={14} className="text-white" />
              </div>
              <div>
                <div className="flex items-center gap-1.5 py-1">
                  {[0, 150, 300].map((delay) => (
                    <span
                      key={delay}
                      className="w-2 h-2 rounded-full animate-bounce"
                      style={{ background: TEAL, animationDelay: `${delay}ms` }}
                    />
                  ))}
                </div>
                {loadingStage === 'thinking' && (
                  <p className="text-xs mt-1" style={{ color: theme.textSecondary }}>
                    Looking up your guidelines...
                  </p>
                )}
                {loadingStage === 'searching' && (
                  <p className="text-xs mt-1" style={{ color: theme.textSecondary }}>
                    Checking safe foods for your stage...
                  </p>
                )}
                {loadingStage === 'slow' && (
                  <p className="text-xs mt-1 animate-pulse" style={{ color: theme.textSecondary }}>
                    AI is taking longer than usual — almost there...
                  </p>
                )}
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input bar */}
        <div
          className="px-6 py-4 flex-shrink-0"
          style={{ borderTop: `1px solid ${borderColor}` }}
        >
          {messages.length === 0 && (
            <div className="flex flex-wrap gap-2 mb-3">
              {SAMPLE_PROMPTS.map((prompt) => (
                <button
                  key={prompt}
                  type="button"
                  onClick={() => void sendMessage(prompt)}
                  disabled={isTyping}
                  className="border border-teal-200 text-teal-700 bg-teal-50 text-sm px-3 py-1.5 rounded-full hover:bg-teal-100 transition-colors disabled:opacity-50"
                >
                  {prompt}
                </button>
              ))}
            </div>
          )}
          <div
            className="flex items-center gap-2 px-4 py-2 rounded-full"
            style={{
              border: `1px solid ${borderColor}`,
              background: isDark ? 'rgba(255,255,255,0.04)' : theme.cardBg,
            }}
          >
            <input
              id="chat-image-upload"
              type="file"
              accept="image/*,.txt,.csv"
              className="hidden"
              onChange={handleFileUpload}
            />
            {voiceSupported && (
              <div className="flex items-center gap-1.5 shrink-0">
                {isListening && (
                  <span className="h-2 w-2 rounded-full bg-red-500 animate-pulse" />
                )}
                <button
                  type="button"
                  onClick={toggleVoice}
                  className="p-1.5 rounded-lg transition-colors hover:opacity-70"
                  style={{ color: isListening ? '#ef4444' : theme.textSecondary }}
                  aria-label="Voice input"
                >
                  {isListening ? <MicOff size={18} /> : <Mic size={18} />}
                </button>
              </div>
            )}
            <button
              type="button"
              onClick={() => document.getElementById('chat-image-upload')?.click()}
              className="p-1.5 rounded-lg text-muted-foreground hover:text-foreground hover:bg-muted/50 transition-colors flex-shrink-0"
              title="Upload image or file"
              aria-label="Upload image or file"
            >
              <Paperclip className="w-4 h-4" />
            </button>
            <input
              type="text"
              className="flex-1 bg-transparent outline-none border-none text-sm min-w-0"
              style={{ color: theme.text }}
              placeholder={isListening ? 'Listening... Speak now' : 'Message GuidaPlate AI...'}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  void sendMessage(input);
                }
              }}
            />
            <button
              type="button"
              onClick={() => void sendMessage(input)}
              disabled={!input.trim() || isTyping}
              className="p-2 rounded-lg shrink-0 transition-opacity disabled:opacity-40"
              style={{
                background: input.trim() ? TEAL : isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.06)',
                color: input.trim() ? '#fff' : theme.textSecondary,
              }}
              aria-label="Send message"
            >
              <SendHorizontal size={18} />
            </button>
          </div>
          <p
            className="text-[11px] text-center mt-2 leading-relaxed"
            style={{ color: theme.textTertiary }}
          >
            AI suggestions can be wrong. Always check with your doctor or dietitian.
          </p>
        </div>
      </main>
      </div>
    </div>
  );
}

export { MealPlanner as MealPlannerPage };
export default MealPlanner;
