import { useState, useRef, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import MapaRestaurantes from "./MapaRestaurantes";

const API_URL = "https://nlp-restaurantes-madrid.onrender.com";

function formatearPrecio(rango) {
  if (!rango) return "";
  const mapa = {
    "euro": "€",
    "euro euro": "€€",
    "euro euro euro": "€€€",
    "euro euro euro euro": "€€€€",
  };
  return mapa[rango.toLowerCase().trim()] || rango;
}

const sugerencias = [
  "Quiero un restaurante peruano",
  "Cena romántica para dos",
  "¿Dónde puedo comer croquetas?",
  "Apto para niños",
  "Mejor relación calidad-precio",
  "Algo con terraza",
  "Cerca de Malasaña",
];

// ── Platos colapsables ───────────────────────────────────────────────────────
function PlatosColapsables({ platos, frecs, renderChip }) {
  const [abierto, setAbierto] = useState(false);
  return (
    <div>
      <button
        onClick={() => setAbierto(v => !v)}
        style={{
          background: "transparent", border: "1px solid #2a2a2a",
          borderRadius: 16, padding: "4px 12px", fontSize: 11,
          color: "#666", cursor: "pointer", marginBottom: abierto ? 8 : 0,
          display: "flex", alignItems: "center", gap: 5,
        }}
      >
        {abierto ? "▲" : "▼"} {abierto ? "Ocultar" : "Otras sugerencias"} ({platos.length})
      </button>
      {abierto && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
          {platos.map((p, i) => renderChip(p, i, false))}
        </div>
      )}
    </div>
  );
}

// ── Vista compacta con tarjetas ──────────────────────────────────────────────
function MensajeCompacto({ texto, onVerDetalle, restaurantesData, consulta, restaurantesDelMensaje }) {
  const lineas = texto.split("\n");
  const bloques = [];
  let intro = [];
  let current = null;

  for (const linea of lineas) {
    const lineaLimpia = linea.replace(/^[\s·\-*•]+/, "").trim();
    // Caso 1: ### Nombre  o  **Nombre** solo en la línea
    const esNombrePuro = lineaLimpia.match(/^#{1,3}\s+(.+)/) || (lineaLimpia.startsWith("**") && lineaLimpia.endsWith("**") && lineaLimpia.length < 60);
    // Caso 2: **Nombre**, datos...  o  · **Nombre**, datos...
    const esNombreConDatos = !esNombrePuro && lineaLimpia.match(/^\*\*(.+?)\*\*[,\s]/);

    if (esNombrePuro || esNombreConDatos) {
      if (current) bloques.push(current);
      let nombre = "";
      if (esNombrePuro) {
        nombre = lineaLimpia.replace(/^#+\s+/, "").replace(/\*\*/g, "").trim();
      } else {
        nombre = esNombreConDatos[1].trim();
      }
      // Buscar distancia en la última línea de intro
      const ultimaIntro = intro.length > 0 ? intro[intro.length - 1] : "";
      const dIntro = ultimaIntro.match(/([\d]+[.,][\d]+)\s*km/i);
      current = { nombre, lineas: [], valoracion: "", precio: "", distancia: dIntro ? dIntro[1] : "" };
      const vInline = lineaLimpia.match(/Valoraci[oó]n[^\d]*([\d.]+)/i) || lineaLimpia.match(/valoraci[oó]n\s+([\d.]+)/i) || lineaLimpia.match(/[\s,]([\d]\.[d]+)/);
      const pInline = lineaLimpia.match(/Rango[^:]*:\s*(€+)/i) || lineaLimpia.match(/(€+)/);
      const dInline = lineaLimpia.match(/([\d]+[.,][\d]+)\s*km/);
      if (vInline) current.valoracion = vInline[1];
      if (pInline) current.precio = pInline[1];
      if (dInline) current.distancia = dInline[1];
    } else if (current) {
      current.lineas.push(linea);
      const vMatch = linea.match(/Valoraci[oó]n[^\d]*([\d.]+)/i) || linea.match(/valoraci[oó]n\s+([\d.]+)/i) || linea.match(/⭐\s*([\d.]+)/);
      const pMatch = linea.match(/Rango[^:]*:\s*(€+)/i) || linea.match(/precio[^:]*:\s*(€+)/i);
      const dMatch = linea.match(/([\d]+[.,][\d]+)\s*km/i);
      if (vMatch && !current.valoracion) current.valoracion = vMatch[1];
      if (pMatch && !current.precio) current.precio = pMatch[1];
      if (dMatch && !current.distancia) current.distancia = dMatch[1];
    } else {
      intro.push(linea);
    }
  }
  if (current) bloques.push(current);

  // Fallback: si no hay bloques pero sí restaurantes en ESTE mensaje, construir tarjetas
  if (bloques.length === 0 && restaurantesDelMensaje && restaurantesDelMensaje.length > 0) {
    const introTextoFallback = texto.split("\n").slice(0, 2).filter(l => l.trim() && !l.includes("**")).join(" ").trim();
    return (
      <div>
        {introTextoFallback && (
          <p style={{ fontSize: 13.5, color: "#ccc", lineHeight: 1.6, marginBottom: 14 }}>{introTextoFallback}</p>
        )}
        {restaurantesDelMensaje.map((datos, i) => {
          const bloqueSintetico = { nombre: datos.nombre, valoracion: datos.valoracion, precio: datos.rango_precio, distancia: datos.distancia_km, lineas: [] };
          return <TarjetaRestaurante key={i} bloque={bloqueSintetico} onVerDetalle={onVerDetalle} restaurantesData={restaurantesData} consulta={consulta} />;
        })}
      </div>
    );
  }

  if (bloques.length === 0) return <MensajeMarkdown texto={texto} />;

  const introTexto = intro.filter(l => l.trim()).join(" ").trim();

  const abrirModal = (bloque) => {
    const key = bloque.nombre.toLowerCase();
    const datos = restaurantesData && restaurantesData[key];
    if (datos) {
      // Parsear platos_frecuencia si viene como string JSON
      let platosFrecuencia = {};
      try {
        platosFrecuencia = datos.platos_frecuencia
          ? (typeof datos.platos_frecuencia === "string"
            ? JSON.parse(datos.platos_frecuencia)
            : datos.platos_frecuencia)
          : {};
      } catch (e) { platosFrecuencia = {}; }

      let perfilCliente = {};
      try {
        perfilCliente = datos.perfil_cliente
          ? (typeof datos.perfil_cliente === "string"
            ? JSON.parse(datos.perfil_cliente)
            : datos.perfil_cliente)
          : {};
      } catch (e) { perfilCliente = {}; }

      onVerDetalle({
        nombre: datos.nombre,
        valoracion: datos.valoracion,
        votaciones: datos.votaciones,
        precio: datos.rango_precio,
        direccion: datos.direccion,
        resumen: datos.resumen,
        positivos: datos.aspectos_positivos || [],
        negativos: datos.aspectos_negativos || [],
        platos: datos.platos_destacados || [],
        platosFrecuencia,
        perfilCliente,
        consulta: consulta || "",
        tokens: datos.tokens || [],
        dato: datos.dato_curioso || "",
        badges: [
          datos.buena_comida && "Buena comida",
          datos.buen_servicio && "Buen servicio",
          datos.buen_ambiente && "Buen ambiente",
          datos.espera_corta && "Servicio rápido",
          datos.buena_relacion_precio_calidad && "Buena relación calidad-precio",
          datos.apto_ninos && "Apto para niños",
          datos.apto_mascotas && "Admite mascotas",
          datos.terraza_exterior && "Terraza exterior",
          datos.recomendable_en_pareja && "Romántico",
          datos.buenas_vistas && "Buenas vistas",
          datos.acceso_minusvalidos && "Accesible",
          datos.buen_postre && "Buenos postres",
          datos.buena_relacion_calidad_precio && "Buena relación calidad-precio",
          datos.apto_grupos && "Apto para grupos",
          datos.opciones_veganas && "Opciones veganas",
          datos.apto_celiaco && "Sin gluten",
        ].filter(Boolean),
        avisos: [
          datos.aviso_espera_larga && "Espera larga",
          datos.aviso_precio_elevado && "Precio elevado",
          datos.aviso_servicio_mejorable && "Servicio mejorable",
        ].filter(Boolean),
        frasesCriterios: datos.frases_criterios || {},
        servicioFrases: datos.servicio_frases || "",
      });
    } else {
      const resumenLinea = bloque.lineas.find(l => l.length > 40 && !l.startsWith("#") && !l.toLowerCase().includes("valoraci"));
      onVerDetalle({
        nombre: bloque.nombre,
        valoracion: bloque.valoracion,
        precio: bloque.precio,
        resumen: resumenLinea || "",
        positivos: [],
        negativos: [],
        platos: [],
        dato: "",
        badges: [],
      });
    }
  };

  return (
    <div>
      {introTexto && (
        <p style={{ fontSize: 13.5, color: "#ccc", marginBottom: 12, lineHeight: 1.6 }}>{introTexto}</p>
      )}
      {bloques.map((bloque, i) => (
        <TarjetaRestaurante key={i} bloque={bloque} onVerDetalle={onVerDetalle} restaurantesData={restaurantesData} consulta={consulta} abrirModal={abrirModal} />
      ))}
    </div>
  );
}

function TarjetaRestaurante({ bloque, onVerDetalle, restaurantesData, consulta, abrirModal }) {
  const palabrasZona = ["cerca", "estoy", "barrio", "zona", "por", "en malasaña", "en chueca", "en lavapiés", "en lavapies", "en salamanca", "en retiro", "en sol", "en chamberi", "en chamberí", "en centro", "en moncloa", "en tetuan", "en tetuán", "en vallecas", "en carabanchel", "en arganzuela", "en latina", "en tetuan"];
  const consultaLower = (consulta || "").toLowerCase();
  const pidioZona = palabrasZona.some(p => consultaLower.includes(p));
  const key = bloque.nombre.toLowerCase();
  const datos = restaurantesData && restaurantesData[key];
  const distancia = datos?.distancia_km || bloque.distancia;

  const handleClick = () => {
    if (abrirModal) {
      abrirModal(bloque);
    } else if (datos) {
      // fallback cuando se llama desde el bloque sintético
      let platosFrecuencia = {};
      try { platosFrecuencia = datos.platos_frecuencia ? JSON.parse(datos.platos_frecuencia) : {}; } catch (e) {}
      let perfilCliente = {};
      try { perfilCliente = datos.perfil_cliente ? JSON.parse(datos.perfil_cliente) : {}; } catch (e) {}
      onVerDetalle({
        nombre: datos.nombre, valoracion: datos.valoracion, votaciones: datos.votaciones,
        precio: datos.rango_precio, direccion: datos.direccion, resumen: datos.resumen,
        positivos: datos.aspectos_positivos || [], negativos: datos.aspectos_negativos || [],
        platos: datos.platos_destacados || [], platosFrecuencia, perfilCliente,
        consulta: consulta || "",
        tokens: datos.tokens || [],
        dato: datos.dato_curioso || "",
        badges: [
          datos.buena_comida && "Buena comida", datos.buen_servicio && "Buen servicio",
          datos.buen_ambiente && "Buen ambiente", datos.espera_corta && "Servicio rápido",
          datos.buena_relacion_precio_calidad && "Buena relación calidad-precio",
          datos.apto_ninos && "Apto para niños", datos.apto_mascotas && "Admite mascotas",
          datos.terraza_exterior && "Terraza exterior", datos.recomendable_en_pareja && "Romántico",
          datos.buenas_vistas && "Buenas vistas", datos.acceso_minusvalidos && "Accesible",
          datos.buen_postre && "Buenos postres",
          datos.buena_relacion_calidad_precio && "Buena relación calidad-precio",
          datos.apto_grupos && "Apto para grupos",
          datos.opciones_veganas && "Opciones veganas",
          datos.apto_celiaco && "Sin gluten",
        ].filter(Boolean),
        avisos: [
          datos.aviso_espera_larga && "Espera larga",
          datos.aviso_precio_elevado && "Precio elevado",
          datos.aviso_servicio_mejorable && "Servicio mejorable",
        ].filter(Boolean),
        frasesCriterios: datos.frases_criterios || {},
        servicioFrases: datos.servicio_frases || "",
      });
    }
  };

  return (
    <div style={{
      background: "#1a1a1a", border: "1px solid #2a2a2a",
      borderRadius: 12, padding: "14px 16px", margin: "8px 0",
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 6 }}>
        <div style={{ fontFamily: "'DM Serif Display', serif", fontSize: 15, color: "#c8a96e" }}>
          {bloque.nombre}
        </div>
        <div style={{ fontSize: 11, color: "#666", flexShrink: 0, marginLeft: 8 }}>
          {bloque.valoracion && "⭐ " + bloque.valoracion}
          {bloque.precio && " · " + formatearPrecio(bloque.precio)}
        </div>
      </div>
      {pidioZona && distancia && (
        <div style={{ fontSize: 11, color: "#555", marginBottom: 6 }}>📍 {distancia} km</div>
      )}
      <button onClick={handleClick} style={{
        background: "transparent", border: "1px solid #2a2a2a",
        color: "#c8a96e", borderRadius: 16, padding: "5px 14px",
        fontSize: 12, cursor: "pointer", fontFamily: "'DM Sans', sans-serif",
      }}>
        Ver detalles →
      </button>
    </div>
  );
}

// ── Parsear markdown ligero ────────────────────────────────────────────────────
function parsearLinea(linea, idx) {
  // Negrita **texto**
  const partes = linea.split(/(\*\*[^*]+\*\*)/g);
  return (
    <span key={idx}>
      {partes.map((p, i) =>
        p.startsWith("**") && p.endsWith("**")
          ? <strong key={i} style={{ color: "#c8a96e", fontWeight: 600 }}>{p.slice(2, -2)}</strong>
          : p
      )}
    </span>
  );
}

function MensajeMarkdown({ texto }) {
  const lineas = texto.split("\n");
  const elementos = [];
  let i = 0;

  while (i < lineas.length) {
    const linea = lineas[i];

    if (!linea.trim()) {
      elementos.push(<div key={i} style={{ height: 8 }} />);
      i++;
      continue;
    }

    // Encabezado ### o ##
    if (linea.startsWith("### ")) {
      elementos.push(
        <p key={i} style={{ fontFamily: "'DM Serif Display', serif", fontSize: 15, color: "#c8a96e", margin: "14px 0 4px", letterSpacing: "-0.2px" }}>
          {linea.slice(4)}
        </p>
      );
      i++;
      continue;
    }
    if (linea.startsWith("## ")) {
      elementos.push(
        <p key={i} style={{ fontFamily: "'DM Serif Display', serif", fontSize: 17, color: "#e8c97e", margin: "16px 0 6px", letterSpacing: "-0.3px" }}>
          {linea.slice(3)}
        </p>
      );
      i++;
      continue;
    }

    // Separador ---
    if (linea.trim() === "---") {
      elementos.push(<hr key={i} style={{ border: "none", borderTop: "1px solid #2a2a2a", margin: "12px 0" }} />);
      i++;
      continue;
    }

    // Lista con - o •
    if (linea.match(/^[-•*]\s/)) {
      const items = [];
      while (i < lineas.length && lineas[i].match(/^[-•*]\s/)) {
        items.push(lineas[i].replace(/^[-•*]\s/, ""));
        i++;
      }
      elementos.push(
        <ul key={`ul-${i}`} style={{ margin: "6px 0", paddingLeft: 0, listStyle: "none", display: "flex", flexDirection: "column", gap: 4 }}>
          {items.map((item, j) => (
            <li key={j} style={{ display: "flex", gap: 8, alignItems: "flex-start", fontSize: 13.5, lineHeight: 1.55, color: "#ccc8c0" }}>
              <span style={{ color: "#c8a96e", flexShrink: 0, marginTop: 2 }}>·</span>
              {parsearLinea(item, j)}
            </li>
          ))}
        </ul>
      );
      continue;
    }

    // Línea con emoji al inicio (✅ ⚠️ 💡) — tratar como bloque especial
    const emojiMatch = linea.match(/^([✅⚠️💡🍽️📍⭐])\s*(.*)/u);
    if (emojiMatch) {
      const colores = { "✅": "#2d6a4f", "⚠️": "#7a5c1e", "💡": "#1e4f6a" };
      const bgColores = { "✅": "#0d2018", "⚠️": "#1e1608", "💡": "#08151e" };
      const emoji = emojiMatch[1];
      elementos.push(
        <div key={i} style={{
          background: bgColores[emoji] || "#1a1a1a",
          border: `1px solid ${colores[emoji] || "#333"}22`,
          borderLeft: `3px solid ${colores[emoji] || "#c8a96e"}`,
          borderRadius: "0 8px 8px 0",
          padding: "8px 12px",
          margin: "6px 0",
          fontSize: 13.5,
          lineHeight: 1.55,
          color: "#ccc8c0",
        }}>
          <span style={{ marginRight: 6 }}>{emoji}</span>
          {parsearLinea(emojiMatch[2], i)}
        </div>
      );
      i++;
      continue;
    }

    // Párrafo normal
    elementos.push(
      <p key={i} style={{ margin: "4px 0", fontSize: 13.5, lineHeight: 1.6, color: "#ccc8c0" }}>
        {parsearLinea(linea, i)}
      </p>
    );
    i++;
  }

  return <div>{elementos}</div>;
}

// ── Animación de entrada ───────────────────────────────────────────────────────
function BurbujaMensaje({ mensaje, index, onVerDetalle, restaurantesData, consulta, restaurantesDelMensaje }) {
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    const t = setTimeout(() => setVisible(true), 40);
    return () => clearTimeout(t);
  }, []);

  const esUsuario = mensaje.rol === "usuario";

  return (
    <div style={{
      display: "flex",
      alignItems: "flex-start",
      gap: 10,
      maxWidth: "88%",
      alignSelf: esUsuario ? "flex-end" : "flex-start",
      flexDirection: esUsuario ? "row-reverse" : "row",
      opacity: visible ? 1 : 0,
      transform: visible ? "translateY(0)" : "translateY(8px)",
      transition: "opacity 0.25s ease, transform 0.25s ease",
    }}>
      {!esUsuario && (
        <div style={{
          width: 30, height: 30, borderRadius: "50%",
          background: "#1e1e1e", border: "1px solid #2a2a2a",
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: 15, flexShrink: 0, marginTop: 2,
        }}>🤖</div>
      )}
      <div style={{
        padding: esUsuario ? "10px 16px" : "14px 16px",
        borderRadius: esUsuario ? "16px 4px 16px 16px" : "4px 16px 16px 16px",
        background: esUsuario ? "#c8a96e" : "#161616",
        color: esUsuario ? "#111" : "#e8e4dc",
        border: esUsuario ? "none" : "1px solid #272727",
        fontSize: 14,
        lineHeight: 1.6,
        maxWidth: "100%",
      }}>
        {esUsuario
          ? <span style={{ fontWeight: 500 }}>{mensaje.texto}</span>
          : <MensajeCompacto texto={mensaje.texto} onVerDetalle={onVerDetalle} restaurantesData={restaurantesData} consulta={consulta} restaurantesDelMensaje={restaurantesDelMensaje} />
        }
      </div>
    </div>
  );
}

// ── Indicador de carga ────────────────────────────────────────────────────────
function IndicadorCarga() {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10, alignSelf: "flex-start" }}>
      <div style={{
        width: 30, height: 30, borderRadius: "50%",
        background: "#1e1e1e", border: "1px solid #2a2a2a",
        display: "flex", alignItems: "center", justifyContent: "center", fontSize: 15,
      }}>🤖</div>
      <div style={{
        background: "#161616", border: "1px solid #272727",
        borderRadius: "4px 16px 16px 16px",
        padding: "14px 18px",
        display: "flex", alignItems: "center", gap: 10,
      }}>
        <div style={{ display: "flex", gap: 5 }}>
          {[0, 0.18, 0.36].map((delay, i) => (
            <span key={i} style={{
              width: 6, height: 6, borderRadius: "50%", background: "#c8a96e",
              display: "inline-block",
              animation: `bounce 1.1s ${delay}s infinite ease-in-out`,
            }} />
          ))}
        </div>
        <span style={{ fontSize: 12, color: "#555", fontStyle: "italic" }}>Consultando restaurantes...</span>
      </div>
    </div>
  );
}

// ── Historial persistente ─────────────────────────────────────────────────────
const HISTORIAL_KEY = "restaurantes_madrid_historial";
const MAX_SESIONES  = 50;

function cargarHistorialGuardado() {
  try { return JSON.parse(localStorage.getItem(HISTORIAL_KEY) || "[]"); }
  catch { return []; }
}

function guardarSesion(sesion) {
  try {
    const h = cargarHistorialGuardado();
    h.unshift(sesion);
    localStorage.setItem(HISTORIAL_KEY, JSON.stringify(h.slice(0, MAX_SESIONES)));
  } catch (e) { console.warn("No se pudo guardar historial:", e); }
}

// ── Panel de historial ────────────────────────────────────────────────────────
function PanelHistorial({ onCerrar, onCargarSesion }) {
  const sesiones = cargarHistorialGuardado();

  const limpiarTodo = () => {
    if (window.confirm("¿Borrar todo el historial?")) {
      localStorage.removeItem(HISTORIAL_KEY);
      onCerrar();
    }
  };

  return (
    <div style={{
      position: "fixed", inset: 0, background: "#000000cc",
      zIndex: 100, display: "flex", justifyContent: "flex-end",
    }} onClick={onCerrar}>
      <div style={{
        width: 340, height: "100%", background: "#0f0f0f",
        borderLeft: "1px solid #2a2a2a", overflowY: "auto",
        padding: "20px 16px", display: "flex", flexDirection: "column", gap: 12,
      }} onClick={e => e.stopPropagation()}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
          <span style={{ fontFamily: "'DM Serif Display', serif", fontSize: 17, color: "#f0ece4" }}>Historial</span>
          <div style={{ display: "flex", gap: 8 }}>
            {sesiones.length > 0 && (
              <button onClick={limpiarTodo} style={{
                background: "transparent", border: "1px solid #3a2a2a",
                color: "#666", borderRadius: 12, padding: "4px 10px", fontSize: 11, cursor: "pointer",
              }}>Borrar todo</button>
            )}
            <button onClick={onCerrar} style={{
              background: "transparent", border: "1px solid #2a2a2a",
              color: "#666", borderRadius: 12, padding: "4px 10px", fontSize: 11, cursor: "pointer",
            }}>✕</button>
          </div>
        </div>
        {sesiones.length === 0 ? (
          <div style={{ color: "#444", fontSize: 13, marginTop: 20, textAlign: "center" }}>
            Aún no hay conversaciones guardadas
          </div>
        ) : sesiones.map((sesion, i) => (
          <div key={i} onClick={() => onCargarSesion(sesion)} style={{
            background: "#161616", border: "1px solid #2a2a2a",
            borderRadius: 10, padding: "12px 14px", cursor: "pointer",
          }}
            onMouseEnter={e => e.currentTarget.style.borderColor = "#c8a96e44"}
            onMouseLeave={e => e.currentTarget.style.borderColor = "#2a2a2a"}
          >
            <div style={{ fontSize: 11, color: "#444", marginBottom: 6 }}>
              {new Date(sesion.fecha).toLocaleString("es-ES", {
                day: "2-digit", month: "2-digit", year: "numeric",
                hour: "2-digit", minute: "2-digit",
              })}
              <span style={{ marginLeft: 8, color: "#333" }}>· {sesion.turnos} pregunta{sesion.turnos !== 1 ? "s" : ""}</span>
            </div>
            {sesion.preguntas.map((p, j) => (
              <div key={j} style={{
                fontSize: 12, color: "#888", marginBottom: 3,
                whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
              }}>
                <span style={{ color: "#c8a96e55", marginRight: 5 }}>→</span>{p}
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}

// ── App principal ─────────────────────────────────────────────────────────────
export default function Chat() {
  const navigate = useNavigate();
  const [mapaRestaurantes, setMapaRestaurantes] = useState([]);
  const [mostrarMapa, setMostrarMapa] = useState(false);
  const [modalRestaurante, setModalRestaurante] = useState(null);
  const [restaurantesData, setRestaurantesData] = useState({});
  const [mostrarHistorial, setMostrarHistorial] = useState(false);
  const preguntasSesionRef = useRef([]);

  const consultaInicial = sessionStorage.getItem("consulta_inicial") || "";
  useEffect(() => {
    if (consultaInicial) {
      sessionStorage.removeItem("consulta_inicial");
      enviar(consultaInicial);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const [mensajes, setMensajes] = useState([{
    rol: "asistente",
    texto: "¡Hola! Soy tu asistente de restaurantes en Madrid.\n¿Qué tipo de restaurante estás buscando hoy?",
  }]);
  const [historial, setHistorial] = useState([]);
  const [input, setInput] = useState("");
  const [cargando, setCargando] = useState(false);
  const [inputFocus, setInputFocus] = useState(false);
  const bottomRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [mensajes, cargando]);

  const enviar = async (texto) => {
    const consulta = (texto || input).trim();
    if (!consulta || cargando) return;
    setInput("");
    preguntasSesionRef.current.push(consulta);
    setMensajes(prev => [...prev, { rol: "usuario", texto: consulta }]);
    setCargando(true);
    try {
      const res = await fetch(`${API_URL}/recomendar`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ consulta, historial }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Error ${res.status}`);
      }
      const data = await res.json();
      const respuesta = data.respuesta || "Sin respuesta del servidor.";
      const consultaActual = data.consulta_usuario || consulta;
      setHistorial(prev => [
        ...prev,
        { role: "user", content: consulta },
        { role: "assistant", content: respuesta },
      ]);
      // Guardar sesión en historial persistente
      guardarSesion({
        fecha: new Date().toISOString(),
        turnos: preguntasSesionRef.current.length,
        preguntas: [...preguntasSesionRef.current],
        conversacion: [
          ...historial,
          { role: "user", content: consulta },
          { role: "assistant", content: respuesta },
        ],
      });
      setMensajes(prev => [...prev, { rol: "asistente", texto: respuesta, consulta: consultaActual, restaurantes: data.restaurantes || [] }]);
      if (data.restaurantes && data.restaurantes.length > 0) {
        setMapaRestaurantes(data.restaurantes);
        // Reemplazar completo — no acumular datos de búsquedas anteriores
        const nuevo = {};
        data.restaurantes.forEach(r => { nuevo[r.nombre.toLowerCase()] = r; });
        setRestaurantesData(nuevo);
      }
    } catch (e) {
      setMensajes(prev => [...prev, {
        rol: "asistente",
        texto: `⚠️ ${e.message || "Error conectando con el servidor. Inténtalo de nuevo."}`,
      }]);
    } finally {
      setCargando(false);
      inputRef.current?.focus();
    }
  };

  const hayConversacion = mensajes.length > 1;

  return (
    <div style={s.root}>
      {/* Header */}
      <header style={s.header}>
        <div style={s.headerInner}>
          <div style={s.logoWrap}>
            <span style={{ fontSize: 26 }}>🍽</span>
          </div>
          <div>
            <div style={s.titulo}>Restaurantes Madrid</div>
            <div style={s.subtitulo}>NLP · nlptown · análisis local</div>
          </div>
        </div>
        <button style={{ ...s.btnNuevo, marginRight: 8 }} onClick={() => navigate("/")}>← Inicio</button>
        {mapaRestaurantes.length > 0 && (
          <button style={{ ...s.btnNuevo, marginRight: 8, borderColor: mostrarMapa ? "#c8a96e" : "#2a2a2a", color: mostrarMapa ? "#c8a96e" : "#666" }}
            onClick={() => setMostrarMapa(m => !m)}>
            {mostrarMapa ? "Ocultar mapa" : "Ver mapa"}
          </button>
        )}
        {hayConversacion && (
          <button
            style={s.btnNuevo}
            onClick={() => { setMensajes([{ rol: "asistente", texto: "¡Hola! ¿Qué tipo de restaurante estás buscando hoy?" }]); setHistorial([]); setMapaRestaurantes([]); setMostrarMapa(false); preguntasSesionRef.current = []; }}
          >
            Nueva búsqueda
          </button>
        )}
      </header>

      {/* Chat */}
      <main style={s.chat}>
        {mensajes.map((m, i) => (
          <BurbujaMensaje key={i} mensaje={m} index={i} onVerDetalle={setModalRestaurante} restaurantesData={restaurantesData} consulta={m.consulta || ""} restaurantesDelMensaje={m.restaurantes || []} />
        ))}
        {cargando && <IndicadorCarga />}
        <div ref={bottomRef} />
      </main>

      {/* Sugerencias */}
      <div style={{ ...s.sugerencias, maxHeight: hayConversacion ? 0 : 120, overflow: "hidden", transition: "max-height 0.4s ease" }}>
        <div style={s.sugerenciasInner}>
          {sugerencias.map((s_, i) => (
            <button key={i} style={s.chip} onClick={() => enviar(s_)}
              onMouseEnter={e => { e.target.style.borderColor = "#c8a96e"; e.target.style.color = "#c8a96e"; }}
              onMouseLeave={e => { e.target.style.borderColor = "#2a2a2a"; e.target.style.color = "#888"; }}
            >
              {s_}
            </button>
          ))}
        </div>
      </div>

      {mostrarMapa && mapaRestaurantes.length > 0 && (
        <div style={{ padding: "0 16px 8px", background: "#0f0f0f" }}>
          <MapaRestaurantes restaurantes={mapaRestaurantes} />
        </div>
      )}

      {/* Input */}
      <footer style={s.footer}>
        <div style={{
          ...s.inputWrap,
          border: inputFocus ? "1px solid #c8a96e55" : "1px solid #272727",
          boxShadow: inputFocus ? "0 0 0 3px #c8a96e11" : "none",
          transition: "border 0.2s, box-shadow 0.2s",
        }}>
          <input
            ref={inputRef}
            style={s.input}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === "Enter" && !e.shiftKey && enviar()}
            onFocus={() => setInputFocus(true)}
            onBlur={() => setInputFocus(false)}
            placeholder="¿Qué tipo de restaurante buscas?"
            disabled={cargando}
            maxLength={300}
          />
          {input.length > 200 && (
            <span style={{ fontSize: 11, color: input.length > 280 ? "#c8a96e" : "#444", flexShrink: 0, marginRight: 4 }}>
              {300 - input.length}
            </span>
          )}
          <button
            style={{ ...s.boton, opacity: cargando || !input.trim() ? 0.35 : 1, transform: input.trim() ? "scale(1)" : "scale(0.95)", transition: "opacity 0.2s, transform 0.2s" }}
            onClick={() => enviar()}
            disabled={cargando || !input.trim()}
            aria-label="Enviar"
          >
            ➤
          </button>
        </div>
        <p style={s.hint}>Puedes preguntar por cocina, ambiente, platos, barrio o nombre del restaurante</p>
      </footer>

      {modalRestaurante && (
        <div style={{
          position: "fixed", inset: 0, zIndex: 1000,
          background: "rgba(0,0,0,0.75)", backdropFilter: "blur(4px)",
          display: "flex", alignItems: "center", justifyContent: "center", padding: 16,
        }} onClick={() => setModalRestaurante(null)}>
          <div style={{
            background: "#161616", border: "1px solid #2a2a2a",
            borderRadius: 16, padding: 24, maxWidth: 480, width: "100%",
            maxHeight: "80vh", overflowY: "auto",
            boxShadow: "0 20px 60px rgba(0,0,0,0.6)",
          }} onClick={e => e.stopPropagation()}>
            {/* Header */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 12 }}>
              <div style={{ flex: 1 }}>
                <div style={{ fontFamily: "'DM Serif Display', serif", fontSize: 20, color: "#f0ece4", marginBottom: 4 }}>
                  {modalRestaurante.nombre}
                </div>
                <div style={{ fontSize: 12, color: "#666", display: "flex", gap: 10, flexWrap: "wrap" }}>
                  {modalRestaurante.valoracion > 0 && <span>⭐ {modalRestaurante.valoracion}{modalRestaurante.votaciones > 0 && ` (${modalRestaurante.votaciones} votos)`}</span>}
                  {modalRestaurante.precio && <span>{formatearPrecio(modalRestaurante.precio)}</span>}
                </div>
                {modalRestaurante.direccion && (
                  <div style={{ fontSize: 11, color: "#444", marginTop: 3 }}>📍 {modalRestaurante.direccion}</div>
                )}
              </div>
              <button onClick={() => setModalRestaurante(null)} style={{
                background: "transparent", border: "none", color: "#555",
                fontSize: 20, cursor: "pointer", padding: "0 4px", flexShrink: 0,
              }}>✕</button>
            </div>

            {/* Badges positivos */}
            {modalRestaurante.badges && modalRestaurante.badges.length > 0 && (
              <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 8 }}>
                {modalRestaurante.badges.map((b, i) => (
                  <span key={i} style={{
                    background: "#0d1f12", border: "1px solid #1a3a20",
                    borderRadius: 12, padding: "3px 10px", fontSize: 11, color: "#4caf82",
                  }}>✓ {b}</span>
                ))}
              </div>
            )}
            {/* Avisos negativos */}
            {modalRestaurante.avisos && modalRestaurante.avisos.length > 0 && (
              <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 14 }}>
                {modalRestaurante.avisos.map((a, i) => (
                  <span key={i} style={{
                    background: "#1f0d0d", border: "1px solid #3a1a1a",
                    borderRadius: 12, padding: "3px 10px", fontSize: 11, color: "#e05555",
                  }}>⚠ {a}</span>
                ))}
              </div>
            )}

            {/* Resumen */}
            {modalRestaurante.resumen && (
              <p style={{ fontSize: 13.5, color: "#ccc", lineHeight: 1.65, marginBottom: 14, fontStyle: "italic" }}>
                "{modalRestaurante.resumen}"
              </p>
            )}

            {/* Lo que dicen los clientes — frases reales de reseñas por criterio */}
            {(() => {
              const ETIQUETAS = {
                ninos: "👶 Apto para niños", mascotas: "🐾 Admite mascotas",
                terraza: "☀️ Terraza", vistas: "🏙️ Vistas", romantico: "🕯️ Romántico",
                musica_directo: "🎵 Música en directo", buen_postre: "🍮 Buenos postres",
                precio_calidad: "💶 Buena relación calidad-precio",
                grupos_grandes: "🎉 Grupos y celebraciones",
                vegano_vegetariano: "🌿 Opciones veganas", sin_gluten: "🌾 Sin gluten",
              };
              const frases = modalRestaurante.frasesCriterios || {};
              const servicioFrases = modalRestaurante.servicioFrases || "";
              const KEYWORDS_CRITERIO = {
                ninos: ["niño","niña","bebé","bebe","peque","familia","infantil","trona","sillita","crío"],
                mascotas: ["perro","mascota","peludo","admiten","dog","pet","can"],
                terraza: ["terraza","exterior","aire libre","patio","velador"],
                vistas: ["vista","panorámica","azotea","rooftop","mirador"],
                musica_directo: ["música","directo","concierto","actuación","jazz","flamenco","en vivo"],
                romantico: ["romántico","íntimo","intimo","romantico","pareja","velas","cena romántica"],
                buen_postre: ["postre","tarta","helado","tiramisú","tiramisu","mousse","brownie","coulant"],
                precio_calidad: ["precio","calidad","económico","asequible","relación","barato"],
                grupos_grandes: ["grupo","celebración","cumpleaños","empresa","evento","varios"],
                vegano_vegetariano: ["vegano","vegana","vegetariano","vegetariana","sin carne","plant"],
                sin_gluten: ["gluten","celiaco","celiaca","celíaco"],
              };
              const entradas = Object.entries(frases).filter(([k, v]) => {
                if (!v || !v.trim() || v.trim().toLowerCase() === "nan" || v.trim().toLowerCase() === "none") return false;
                const keywords = KEYWORDS_CRITERIO[k];
                if (!keywords) return true;
                const textoLower = v.toLowerCase();
                return keywords.some(kw => textoLower.includes(kw));
              });
              if (entradas.length === 0 && !servicioFrases) return null;
              return (
                <div style={{ marginBottom: 14 }}>
                  <div style={{ fontSize: 11, color: "#c8a96e", fontWeight: 600,
                    letterSpacing: "1px", textTransform: "uppercase", marginBottom: 10 }}>
                    Lo que dicen los clientes
                  </div>
                  {entradas.map(([criterio, texto]) => (
                    <div key={criterio} style={{ marginBottom: 8,
                      background: "#111", borderRadius: 8, padding: "8px 12px",
                      borderLeft: "2px solid #1a3a20" }}>
                      <div style={{ fontSize: 11, color: "#4caf82", fontWeight: 600, marginBottom: 4 }}>
                        {ETIQUETAS[criterio] || criterio}
                      </div>
                      {texto.split("|").slice(0, 2).map((frase, i) => (
                        <div key={i} style={{ fontSize: 12, color: "#888",
                          fontStyle: "italic", lineHeight: 1.5 }}>
                          "{frase.trim().substring(0, 120)}"
                        </div>
                      ))}
                    </div>
                  ))}
                  {servicioFrases && (
                    <div style={{ marginBottom: 8,
                      background: "#111", borderRadius: 8, padding: "8px 12px",
                      borderLeft: "2px solid #1a3a20" }}>
                      <div style={{ fontSize: 11, color: "#4caf82", fontWeight: 600, marginBottom: 4 }}>
                        💬 Servicio
                      </div>
                      {servicioFrases.split("|").slice(0, 2).map((frase, i) => (
                        <div key={i} style={{ fontSize: 12, color: "#888",
                          fontStyle: "italic", lineHeight: 1.5 }}>
                          "{frase.trim().substring(0, 120)}"
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              );
            })()}

            {/* Aspectos positivos */}
            {modalRestaurante.positivos && modalRestaurante.positivos.length > 0 && (
              <div style={{ background: "#0d2018", borderLeft: "3px solid #2d6a4f", borderRadius: "0 8px 8px 0", padding: "10px 14px", marginBottom: 10 }}>
                <div style={{ fontSize: 12, color: "#5a9a70", fontWeight: 500, marginBottom: 6 }}>✅ Lo que destacan los clientes</div>
                {modalRestaurante.positivos.map((p, i) => (
                  <div key={i} style={{ fontSize: 13, color: "#aaa", marginBottom: 3 }}>· {p}</div>
                ))}
              </div>
            )}

            {/* Platos */}
            {modalRestaurante.platos && modalRestaurante.platos.length > 0 && (() => {
              const frecs = modalRestaurante.platosFrecuencia || {};
              const stopwords = ["quiero","comer","busco","estoy","en","cerca","de","un","una","los","las","donde","puedo","hay","con","para","que","me","ir","a","restaurante","haya","buen","buena","buenos"];
              // Usar tokens del backend si están disponibles, si no extraer de la consulta
              const palabrasConsulta = (modalRestaurante.tokens && modalRestaurante.tokens.length > 0)
                ? modalRestaurante.tokens.map(t => t.toLowerCase())
                : (modalRestaurante.consulta || "").toLowerCase().split(/\s+/)
                    .filter(p => p.length > 3 && !stopwords.includes(p));

              const platosOrdenados = [...modalRestaurante.platos].sort((a, b) => {
                const aEsBuscado = palabrasConsulta.some(p => a.toLowerCase().includes(p));
                const bEsBuscado = palabrasConsulta.some(p => b.toLowerCase().includes(p));
                if (aEsBuscado && !bEsBuscado) return -1;
                if (!aEsBuscado && bEsBuscado) return 1;
                const keyA = Object.keys(frecs).find(k => k.toLowerCase().includes(a.toLowerCase()) || a.toLowerCase().includes(k.toLowerCase()));
                const keyB = Object.keys(frecs).find(k => k.toLowerCase().includes(b.toLowerCase()) || b.toLowerCase().includes(k.toLowerCase()));
                return (frecs[keyB] || 0) - (frecs[keyA] || 0);
              });

              const platosBuscados = platosOrdenados.filter(p =>
                palabrasConsulta.some(w => p.toLowerCase().includes(w))
              );
              const platosResto = platosOrdenados.filter(p =>
                !palabrasConsulta.some(w => p.toLowerCase().includes(w))
              );
              const hayBuscados = platosBuscados.length > 0;

              const renderChip = (p, i, destacado) => {
                const key = Object.keys(frecs).find(k => k.toLowerCase().includes(p.toLowerCase()) || p.toLowerCase().includes(k.toLowerCase()));
                const n = key ? frecs[key] : null;
                return (
                  <span key={i} style={{
                    background: destacado ? "#2a1f00" : "#111",
                    border: destacado ? "1px solid #c8a96e88" : "1px solid #2a2a2a",
                    borderRadius: 10, padding: "4px 12px", fontSize: destacado ? 13 : 12,
                    color: destacado ? "#c8a96e" : "#888",
                    fontWeight: destacado ? 500 : 400,
                  }}>
                    {p}
                    {n ? <span style={{ color: destacado ? "#c8a96eaa" : "#c8a96e55", fontSize: 11, marginLeft: 4 }}>({n}/90 reseñas)</span> : null}
                  </span>
                );
              };

              return (
                <div style={{ background: "#1a1505", borderLeft: "3px solid #c8a96e44", borderRadius: "0 8px 8px 0", padding: "10px 14px", marginBottom: 10 }}>
                  {hayBuscados ? (
                    <>
                      <div style={{ fontSize: 12, color: "#c8a96e", fontWeight: 500, marginBottom: 8 }}>🍽 Plato buscado</div>
                      <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: platosResto.length > 0 ? 10 : 0 }}>
                        {platosBuscados.map((p, i) => renderChip(p, i, true))}
                      </div>
                      {platosResto.length > 0 && (
                        <PlatosColapsables platos={platosResto} frecs={frecs} renderChip={renderChip} />
                      )}
                    </>
                  ) : (
                    <>
                      <div style={{ fontSize: 12, color: "#c8a96e", fontWeight: 500, marginBottom: 6 }}>🍽 Platos recomendados</div>
                      <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                        {platosOrdenados.map((p, i) => renderChip(p, i, false))}
                      </div>
                    </>
                  )}
                </div>
              );
            })()}

            {/* Perfil de cliente — basado en análisis NLP de reseñas reales */}
            {modalRestaurante.perfilCliente && Object.keys(modalRestaurante.perfilCliente.perfiles || {}).length > 0 && (() => {
              const perfiles  = modalRestaurante.perfilCliente.perfiles  || {};
              const momentos  = modalRestaurante.perfilCliente.momentos  || {};
              const iconos    = { familia: "👨‍👩‍👧", pareja: "💑", amigos: "👥", empresa: "💼", turista: "🌍", solo: "🧍" };
              const iconosMom = { comida: "☀️", cena: "🌙", brunch: "☕" };
              return (
                <div style={{ background: "#0e0e1a", borderLeft: "3px solid #4a4a8a", borderRadius: "0 8px 8px 0", padding: "10px 14px", marginBottom: 10 }}>
                  <div style={{ fontSize: 12, color: "#8888cc", fontWeight: 500, marginBottom: 8 }}>👤 Perfil de clientes — basado en reseñas reales</div>
                  {/* Tipos de visita */}
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 8 }}>
                    {Object.entries(perfiles).map(([tipo, pct], i) => (
                      <div key={i} style={{
                        background: "#16162a", border: "1px solid #2a2a4a",
                        borderRadius: 10, padding: "4px 10px",
                        display: "flex", alignItems: "center", gap: 5,
                      }}>
                        <span style={{ fontSize: 14 }}>{iconos[tipo] || "👤"}</span>
                        <span style={{ fontSize: 12, color: "#aaa" }}>{tipo.charAt(0).toUpperCase() + tipo.slice(1)}</span>
                        <span style={{ fontSize: 11, color: "#8888cc", fontWeight: 600 }}>{Math.round(pct * 100)}%</span>
                      </div>
                    ))}
                  </div>
                  {/* Momentos del día */}
                  {Object.keys(momentos).length > 0 && (
                    <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                      {Object.entries(momentos).map(([momento, pct], i) => (
                        <div key={i} style={{ fontSize: 11, color: "#666", display: "flex", alignItems: "center", gap: 3 }}>
                          <span>{iconosMom[momento] || "🕐"}</span>
                          <span style={{ color: "#555" }}>{momento.charAt(0).toUpperCase() + momento.slice(1)}: </span>
                          <span style={{ color: "#8888cc" }}>{Math.round(pct * 100)}%</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              );
            })()}

            {/* Aspectos negativos */}
            {modalRestaurante.negativos && modalRestaurante.negativos.length > 0 && (
              <div style={{ background: "#1e1608", borderLeft: "3px solid #7a5c1e", borderRadius: "0 8px 8px 0", padding: "10px 14px", marginBottom: 10 }}>
                <div style={{ fontSize: 12, color: "#c8963e", fontWeight: 500, marginBottom: 6 }}>⚠️ A tener en cuenta</div>
                {modalRestaurante.negativos.map((n, i) => (
                  <div key={i} style={{ fontSize: 13, color: "#aaa", marginBottom: 3 }}>· {n}</div>
                ))}
              </div>
            )}

            {/* Dato curioso */}
            {modalRestaurante.dato && (
              <div style={{ background: "#08151e", borderLeft: "3px solid #1e4f6a", borderRadius: "0 8px 8px 0", padding: "10px 14px", marginTop: 4 }}>
                <div style={{ fontSize: 12, color: "#5a8aaa", fontWeight: 500, marginBottom: 4 }}>💡 Dato curioso</div>
                <div style={{ fontSize: 13, color: "#aaa" }}>{modalRestaurante.dato}</div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Panel historial */}
      {mostrarHistorial && (
        <PanelHistorial
          onCerrar={() => setMostrarHistorial(false)}
          onCargarSesion={(sesion) => {
            setMensajes([
              { rol: "asistente", texto: "¡Hola! Soy tu asistente de restaurantes en Madrid.\n¿Qué tipo de restaurante estás buscando hoy?" },
              ...sesion.conversacion.map((m, i) => ({
                rol: m.role === "user" ? "usuario" : "asistente",
                texto: m.content,
              }))
            ]);
            setHistorial(sesion.conversacion);
            preguntasSesionRef.current = [...sesion.preguntas];
            setMostrarHistorial(false);
          }}
        />
      )}

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:ital,wght@0,400;0,500;1,400&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #0c0c0c; font-family: 'DM Sans', sans-serif; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #2a2a2a; border-radius: 4px; }
        @keyframes bounce {
          0%, 80%, 100% { transform: translateY(0); opacity: 0.5; }
          40% { transform: translateY(-5px); opacity: 1; }
        }
      `}</style>
    </div>
  );
}

const s = {
  root: {
    display: "flex",
    flexDirection: "column",
    height: "100vh",
    maxWidth: 700,
    margin: "0 auto",
    background: "#0f0f0f",
    color: "#f0ece4",
  },
  header: {
    padding: "14px 20px",
    borderBottom: "1px solid #1e1e1e",
    background: "#0f0f0f",
    position: "sticky",
    top: 0,
    zIndex: 10,
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
  },
  headerInner: { display: "flex", alignItems: "center", gap: 12 },
  logoWrap: {
    width: 42, height: 42, borderRadius: 10,
    background: "#1a1505", border: "1px solid #c8a96e22",
    display: "flex", alignItems: "center", justifyContent: "center",
  },
  titulo: {
    fontFamily: "'DM Serif Display', serif",
    fontSize: 18, color: "#f0ece4", letterSpacing: "-0.3px",
  },
  subtitulo: {
    fontSize: 10, color: "#444", letterSpacing: "0.8px",
    textTransform: "uppercase", marginTop: 2,
  },
  btnNuevo: {
    background: "transparent", border: "1px solid #2a2a2a",
    color: "#666", borderRadius: 20, padding: "5px 14px",
    fontSize: 12, cursor: "pointer",
  },
  chat: {
    flex: 1, overflowY: "auto", padding: "20px 16px",
    display: "flex", flexDirection: "column", gap: 14,
  },
  sugerencias: { background: "#0f0f0f" },
  sugerenciasInner: {
    padding: "0 16px 12px",
    display: "flex", flexWrap: "wrap", gap: 7,
  },
  chip: {
    background: "transparent", border: "1px solid #2a2a2a",
    color: "#888", borderRadius: 20, padding: "6px 14px",
    fontSize: 12, cursor: "pointer",
  },
  footer: {
    padding: "10px 16px 16px",
    borderTop: "1px solid #1e1e1e",
    background: "#0f0f0f",
  },
  inputWrap: {
    display: "flex", gap: 8, alignItems: "center",
    background: "#161616", borderRadius: 28,
    padding: "4px 4px 4px 16px",
  },
  input: {
    flex: 1, background: "transparent", border: "none",
    color: "#f0ece4", fontSize: 14, outline: "none",
    padding: "8px 0", fontFamily: "'DM Sans', sans-serif",
  },
  boton: {
    background: "#c8a96e", border: "none", borderRadius: "50%",
    width: 38, height: 38, fontSize: 14, cursor: "pointer",
    color: "#111", fontWeight: "bold", flexShrink: 0,
    display: "flex", alignItems: "center", justifyContent: "center",
  },
  hint: {
    fontSize: 11, color: "#333", textAlign: "center",
    marginTop: 8, letterSpacing: "0.2px",
  },
};
