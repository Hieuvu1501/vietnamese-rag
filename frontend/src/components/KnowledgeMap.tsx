import {
  forceCollide,
  forceLink,
  forceManyBody,
  forceSimulation,
  forceX,
  forceY,
  type SimulationLinkDatum,
  type SimulationNodeDatum,
} from 'd3-force'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { getGraph } from '../api'
import type { GraphData, GraphEdge, GraphNode } from '../types'

const DOC_COLORS = ['#3b82f6', '#f59e0b', '#10b981', '#ef4444', '#a855f7', '#06b6d4', '#ec4899', '#84cc16']
const ALWAYS_LABELED = 25

type Point = { x: number; y: number }
type SimNode = GraphNode & SimulationNodeDatum
// scale = SVG units per screen pixel, so labels can stay a constant on-screen size at any zoom.
type View = { cx: number; cy: number; scale: number }

function nodeRadius(degree: number) {
  return 4 + Math.sqrt(degree) * 2.5
}

function computeLayout(nodes: GraphNode[], edges: GraphEdge[]): Map<string, Point> {
  const simNodes: SimNode[] = nodes.map((n) => ({ ...n }))
  const simLinks: SimulationLinkDatum<SimNode>[] = edges.map((e) => ({ source: e.source, target: e.target }))
  forceSimulation<SimNode>(simNodes)
    .force('link', forceLink<SimNode, SimulationLinkDatum<SimNode>>(simLinks).id((d) => d.id).distance(60))
    .force('charge', forceManyBody<SimNode>().strength(-110))
    // Weak pull to the centre keeps disconnected clusters from drifting off-screen.
    .force('x', forceX<SimNode>(0).strength(0.04))
    .force('y', forceY<SimNode>(0).strength(0.04))
    .force('collide', forceCollide<SimNode>((d) => nodeRadius(d.degree) + 3))
    .stop()
    .tick(300)
  return new Map(simNodes.map((n) => [n.id, { x: n.x ?? 0, y: n.y ?? 0 }]))
}

function fitView(positions: Map<string, Point>, width: number, height: number, padding = 120): View {
  if (positions.size === 0 || width === 0 || height === 0) return { cx: 0, cy: 0, scale: 1 }
  const xs = [...positions.values()].map((p) => p.x)
  const ys = [...positions.values()].map((p) => p.y)
  const [minX, maxX, minY, maxY] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)]
  const scale = Math.max((maxX - minX + padding) / width, (maxY - minY + padding) / height, 0.2)
  return { cx: (minX + maxX) / 2, cy: (minY + maxY) / 2, scale }
}

function norm(s: string) {
  return s.normalize('NFC').toLowerCase()
}

export default function KnowledgeMap() {
  const [data, setData] = useState<GraphData | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [docFilter, setDocFilter] = useState<number | 'all'>('all')
  const [search, setSearch] = useState('')
  const [selected, setSelected] = useState<string | null>(null)
  const [size, setSize] = useState({ width: 0, height: 0 })
  const [view, setView] = useState<View>({ cx: 0, cy: 0, scale: 1 })
  const containerRef = useRef<HTMLDivElement>(null)
  const svgRef = useRef<SVGSVGElement>(null)
  const drag = useRef<{ x: number; y: number; moved: boolean; nodeId: string | null } | null>(null)

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      setData(await getGraph())
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    load()
  }, [load])

  const { nodes, edges } = useMemo(() => {
    if (!data) return { nodes: [] as GraphNode[], edges: [] as GraphEdge[] }
    const edges = docFilter === 'all' ? data.edges : data.edges.filter((e) => e.document_ids.includes(docFilter))
    const degree = new Map<string, number>()
    for (const e of edges) {
      degree.set(e.source, (degree.get(e.source) ?? 0) + 1)
      degree.set(e.target, (degree.get(e.target) ?? 0) + 1)
    }
    const nodes = data.nodes.filter((n) => degree.has(n.id)).map((n) => ({ ...n, degree: degree.get(n.id) ?? 0 }))
    return { nodes, edges }
  }, [data, docFilter])

  const positions = useMemo(() => computeLayout(nodes, edges), [nodes, edges])
  const nodeById = useMemo(() => new Map(nodes.map((n) => [n.id, n])), [nodes])
  const docIndex = useMemo(() => new Map((data?.documents ?? []).map((d, i) => [d.id, i])), [data])
  const docName = useMemo(() => new Map((data?.documents ?? []).map((d) => [d.id, d.filename])), [data])

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const observer = new ResizeObserver(([entry]) =>
      setSize({ width: entry.contentRect.width, height: entry.contentRect.height }),
    )
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  // Re-fit when the layout changes or the viewport is first measured - not on every resize, or opening the
  // side panel would zoom the view out from under the user. (Adjust-state-during-render, not an effect.)
  const [fittedFor, setFittedFor] = useState<{ positions: Map<string, Point>; measured: boolean } | null>(null)
  const measured = size.width > 0 && size.height > 0
  if (fittedFor?.positions !== positions || fittedFor.measured !== measured) {
    setFittedFor({ positions, measured })
    setView(fitView(positions, size.width, size.height))
  }

  // For picks from the search list / side panel, where the entity may be off-screen: zoom to fit its
  // neighbourhood. Plain map clicks don't move the view.
  const focusNode = (id: string) => {
    setSelected(id)
    const hood = new Map<string, Point>()
    for (const e of edges) {
      if (e.source !== id && e.target !== id) continue
      for (const nid of [e.source, e.target]) {
        const p = positions.get(nid)
        if (p) hood.set(nid, p)
      }
    }
    const self = positions.get(id)
    if (self) hood.set(id, self)
    // The side panel (w-80) opens on first selection and takes that much width from the map.
    const width = size.width - (selectedNode ? 0 : 320)
    if (hood.size > 0) setView(fitView(hood, width, size.height, 40))
  }

  // Native listener: React registers wheel handlers as passive, so preventDefault() there can't stop page scroll.
  useEffect(() => {
    const svg = svgRef.current
    if (!svg) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const rect = svg.getBoundingClientRect()
      const mx = e.clientX - rect.left
      const my = e.clientY - rect.top
      setView((v) => {
        const scale = Math.min(20, Math.max(0.02, v.scale * Math.exp(e.deltaY * 0.0015)))
        const px = v.cx - (rect.width * v.scale) / 2 + mx * v.scale
        const py = v.cy - (rect.height * v.scale) / 2 + my * v.scale
        return { scale, cx: px + (rect.width * scale) / 2 - mx * scale, cy: py + (rect.height * scale) / 2 - my * scale }
      })
    }
    svg.addEventListener('wheel', onWheel, { passive: false })
    return () => svg.removeEventListener('wheel', onWheel)
  }, [])

  const selectedNode = selected ? nodeById.get(selected) ?? null : null

  const neighbors = useMemo(() => {
    if (!selectedNode) return null
    const set = new Set([selectedNode.id])
    for (const e of edges) {
      if (e.source === selectedNode.id) set.add(e.target)
      if (e.target === selectedNode.id) set.add(e.source)
    }
    return set
  }, [selectedNode, edges])

  const query = norm(search.trim())
  const matches = useMemo(() => (query ? nodes.filter((n) => norm(n.label).includes(query)) : []), [nodes, query])
  const matchIds = useMemo(() => new Set(matches.map((n) => n.id)), [matches])

  const labeled = useMemo(() => {
    const set = new Set(
      [...nodes]
        .sort((a, b) => b.degree - a.degree)
        .slice(0, ALWAYS_LABELED)
        .map((n) => n.id),
    )
    neighbors?.forEach((id) => set.add(id))
    matchIds.forEach((id) => set.add(id))
    return set
  }, [nodes, neighbors, matchIds])

  const colorOf = useCallback(
    (n: GraphNode) => DOC_COLORS[(docIndex.get(n.document_ids[0]) ?? 0) % DOC_COLORS.length],
    [docIndex],
  )

  const edgeLayer = useMemo(
    () =>
      edges.map((e, i) => {
        const s = positions.get(e.source)
        const t = positions.get(e.target)
        if (!s || !t) return null
        const hl = !!selectedNode && (e.source === selectedNode.id || e.target === selectedNode.id)
        // Stop the line at the target's rim so the arrowhead isn't hidden under the circle.
        const dx = t.x - s.x
        const dy = t.y - s.y
        const len = Math.hypot(dx, dy) || 1
        const inset = nodeRadius(nodeById.get(e.target)?.degree ?? 0) + 2
        return (
          <line
            key={i}
            x1={s.x}
            y1={s.y}
            x2={t.x - (dx / len) * inset}
            y2={t.y - (dy / len) * inset}
            vectorEffect="non-scaling-stroke"
            strokeWidth={hl ? 2 : 1}
            opacity={neighbors ? (hl ? 0.95 : 0.06) : 0.35}
            markerEnd={hl ? 'url(#map-arrow)' : undefined}
            className={hl ? 'stroke-blue-500' : 'stroke-neutral-400 dark:stroke-neutral-500'}
          />
        )
      }),
    [edges, positions, selectedNode, neighbors, nodeById],
  )

  const nodeLayer = useMemo(
    () =>
      nodes.map((n) => {
        const p = positions.get(n.id)
        if (!p) return null
        const r = nodeRadius(n.degree)
        const dim = neighbors && !neighbors.has(n.id)
        return (
          <g key={n.id} data-node={n.id} opacity={dim ? 0.15 : 1} className="cursor-pointer">
            {matchIds.has(n.id) && (
              <circle cx={p.x} cy={p.y} r={r + 5} fill="none" strokeWidth={2} vectorEffect="non-scaling-stroke" className="stroke-amber-400" />
            )}
            <circle
              cx={p.x}
              cy={p.y}
              r={r}
              fill={colorOf(n)}
              strokeWidth={n.id === selectedNode?.id ? 3 : n.document_ids.length > 1 ? 1.5 : 0}
              vectorEffect="non-scaling-stroke"
              className={n.id === selectedNode?.id ? 'stroke-blue-300' : 'stroke-white dark:stroke-neutral-100'}
            >
              <title>{n.label}</title>
            </circle>
          </g>
        )
      }),
    [nodes, positions, neighbors, matchIds, selectedNode, colorOf],
  )

  const s = view.scale
  // Hubs (e.g. one person with 28 relationships) ring themselves with long names; keep neighbours short.
  const maxLabel = selectedNode ? 20 : 40
  const labelLayer = nodes.map((n) => {
    const p = positions.get(n.id)
    if (!p || !labeled.has(n.id) || (neighbors && !neighbors.has(n.id))) return null
    const limit = n.id === selectedNode?.id ? 60 : maxLabel
    return (
      <text
        key={n.id}
        data-node={n.id}
        x={p.x}
        y={p.y - nodeRadius(n.degree) - 4 * s}
        fontSize={11 * s}
        textAnchor="middle"
        strokeWidth={3 * s}
        paintOrder="stroke"
        className="cursor-pointer fill-neutral-800 stroke-white dark:fill-neutral-100 dark:stroke-neutral-900"
      >
        {n.label.length > limit ? `${n.label.slice(0, limit - 1)}…` : n.label}
        <title>{n.label}</title>
      </text>
    )
  })

  const facts = selectedNode ? edges.filter((e) => e.source === selectedNode.id || e.target === selectedNode.id) : []

  // Beyond ~12 edges the midpoint labels just overlap each other; the side panel lists them instead.
  const predicateLayer =
    selectedNode &&
    facts.length <= 12 &&
    edges.map((e, i) => {
      if (e.source !== selectedNode.id && e.target !== selectedNode.id) return null
      const a = positions.get(e.source)
      const b = positions.get(e.target)
      if (!a || !b) return null
      return (
        <text
          key={i}
          x={(a.x + b.x) / 2}
          y={(a.y + b.y) / 2}
          fontSize={10 * s}
          textAnchor="middle"
          strokeWidth={3 * s}
          paintOrder="stroke"
          className="pointer-events-none fill-blue-600 stroke-white dark:fill-blue-300 dark:stroke-neutral-900"
        >
          {e.predicate}
        </text>
      )
    })

  const labelOf = (id: string) => nodeById.get(id)?.label ?? id
  const isEmpty = !loading && !error && (data?.nodes.length ?? 0) === 0

  return (
    <div className="flex h-full min-h-0 flex-1 flex-col bg-white dark:bg-neutral-900">
      <div className="flex flex-wrap items-center gap-3 border-b border-neutral-200 px-4 py-2 dark:border-neutral-800">
        <span className="text-xs text-neutral-500 dark:text-neutral-400">
          {nodes.length} entities · {edges.length} relationships
        </span>
        <input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Tìm thực thể…"
          className="w-56 rounded-md border border-neutral-300 px-2 py-1 text-sm outline-none focus:border-blue-400 dark:border-neutral-700 dark:bg-neutral-800 dark:text-neutral-100"
        />
        {query && <span className="text-xs text-neutral-500 dark:text-neutral-400">{matches.length} match(es)</span>}
        <div className="ml-auto flex gap-2">
          <button
            onClick={() => setView(fitView(positions, size.width, size.height))}
            className="rounded-md border border-neutral-300 px-2 py-1 text-xs text-neutral-700 hover:bg-neutral-100 dark:border-neutral-700 dark:text-neutral-300 dark:hover:bg-neutral-800"
          >
            Fit
          </button>
          <button
            onClick={load}
            className="rounded-md border border-neutral-300 px-2 py-1 text-xs text-neutral-700 hover:bg-neutral-100 dark:border-neutral-700 dark:text-neutral-300 dark:hover:bg-neutral-800"
          >
            Refresh
          </button>
        </div>
      </div>

      {data && data.documents.length > 0 && (
        <div className="flex flex-wrap items-center gap-1.5 border-b border-neutral-200 px-4 py-2 dark:border-neutral-800">
          <button
            onClick={() => setDocFilter('all')}
            className={`rounded-full px-2 py-0.5 text-xs ${docFilter === 'all' ? 'bg-neutral-900 text-white dark:bg-neutral-100 dark:text-neutral-900' : 'text-neutral-600 hover:bg-neutral-100 dark:text-neutral-300 dark:hover:bg-neutral-800'}`}
          >
            All documents
          </button>
          {data.documents.map((d, i) => (
            <button
              key={d.id}
              onClick={() => setDocFilter(d.id)}
              title={d.filename}
              className={`flex max-w-[16rem] items-center gap-1.5 rounded-full px-2 py-0.5 text-xs ${docFilter === d.id ? 'bg-neutral-900 text-white dark:bg-neutral-100 dark:text-neutral-900' : 'text-neutral-600 hover:bg-neutral-100 dark:text-neutral-300 dark:hover:bg-neutral-800'}`}
            >
              <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: DOC_COLORS[i % DOC_COLORS.length] }} />
              <span className="truncate">
                #{d.id} {d.filename}
              </span>
            </button>
          ))}
          <span className="ml-auto text-xs text-neutral-400">white ring = entity appears in more than one document</span>
        </div>
      )}

      <div className="flex min-h-0 flex-1">
        <div ref={containerRef} className="relative min-w-0 flex-1">
          <svg
            ref={svgRef}
            className="absolute inset-0 h-full w-full cursor-grab touch-none select-none active:cursor-grabbing"
            viewBox={`${view.cx - (size.width * s) / 2} ${view.cy - (size.height * s) / 2} ${size.width * s || 1} ${size.height * s || 1}`}
            onPointerDown={(e) => {
              const nodeId = (e.target as Element).closest('[data-node]')?.getAttribute('data-node') ?? null
              drag.current = { x: e.clientX, y: e.clientY, moved: false, nodeId }
              e.currentTarget.setPointerCapture(e.pointerId)
            }}
            onPointerMove={(e) => {
              const d = drag.current
              if (!d) return
              const dx = e.clientX - d.x
              const dy = e.clientY - d.y
              if (!d.moved && Math.abs(dx) + Math.abs(dy) < 4) return
              d.moved = true
              d.x = e.clientX
              d.y = e.clientY
              setView((v) => ({ ...v, cx: v.cx - dx * v.scale, cy: v.cy - dy * v.scale }))
            }}
            onPointerUp={() => {
              const d = drag.current
              drag.current = null
              if (d && !d.moved) setSelected(d.nodeId)
            }}
          >
            <defs>
              <marker id="map-arrow" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="7" markerHeight="7" orient="auto">
                <path d="M0,0 L10,5 L0,10 z" className="fill-blue-500" />
              </marker>
            </defs>
            <g>{edgeLayer}</g>
            <g>{nodeLayer}</g>
            <g>{labelLayer}</g>
            <g>{predicateLayer}</g>
          </svg>

          {loading && !data && (
            <div className="absolute inset-0 flex items-center justify-center text-sm text-neutral-400">Loading graph…</div>
          )}
          {error && (
            <div className="absolute inset-0 flex items-center justify-center text-sm text-red-500">
              Could not load the graph: {error}
            </div>
          )}
          {isEmpty && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-1 text-center text-sm text-neutral-500 dark:text-neutral-400">
              <p className="font-medium">No graph facts yet</p>
              <p className="max-w-sm">
                Facts are extracted from documents uploaded while ENABLE_GRAPH_RAG=true in backend/.env.
              </p>
            </div>
          )}
          {query && matches.length > 0 && !selectedNode && (
            <div className="absolute top-3 left-3 max-h-64 w-64 overflow-y-auto rounded-lg border border-neutral-200 bg-white/95 p-1 text-sm shadow-sm dark:border-neutral-700 dark:bg-neutral-800/95">
              {matches.slice(0, 20).map((n) => (
                <button
                  key={n.id}
                  onClick={() => focusNode(n.id)}
                  className="block w-full truncate rounded px-2 py-1 text-left text-neutral-700 hover:bg-neutral-100 dark:text-neutral-200 dark:hover:bg-neutral-700"
                >
                  {n.label}
                </button>
              ))}
            </div>
          )}
        </div>

        {selectedNode && (
          <aside className="flex w-80 shrink-0 flex-col border-l border-neutral-200 bg-neutral-50 dark:border-neutral-800 dark:bg-neutral-950">
            <div className="flex items-start justify-between gap-2 border-b border-neutral-200 p-4 dark:border-neutral-800">
              <div className="min-w-0">
                <h2 className="font-semibold break-words text-neutral-900 dark:text-neutral-100">{selectedNode.label}</h2>
                <p className="mt-1 text-xs text-neutral-500 dark:text-neutral-400">
                  {facts.length} relationship{facts.length === 1 ? '' : 's'} ·{' '}
                  {selectedNode.document_ids.map((id) => `#${id}`).join(', ')}
                </p>
              </div>
              <button
                onClick={() => setSelected(null)}
                aria-label="Close"
                className="rounded px-1.5 text-neutral-400 hover:bg-neutral-200 hover:text-neutral-700 dark:hover:bg-neutral-800 dark:hover:text-neutral-200"
              >
                ✕
              </button>
            </div>
            <ul className="flex-1 space-y-2 overflow-y-auto p-4 text-sm">
              {facts.map((e, i) => (
                <li key={i} className="rounded bg-white p-2 dark:bg-neutral-900">
                  <div className="text-neutral-800 dark:text-neutral-100">
                    <EntityLink id={e.source} label={labelOf(e.source)} current={selectedNode.id} onSelect={focusNode} />{' '}
                    <span className="text-blue-600 dark:text-blue-300">{e.predicate}</span>{' '}
                    <EntityLink id={e.target} label={labelOf(e.target)} current={selectedNode.id} onSelect={focusNode} />
                    {e.count > 1 && <span className="ml-1 text-xs text-neutral-400">×{e.count}</span>}
                  </div>
                  <div className="mt-1 truncate text-xs text-neutral-400" title={e.document_ids.map((id) => docName.get(id)).join('\n')}>
                    {e.document_ids.map((id) => `#${id} ${docName.get(id) ?? ''}`).join(' · ')}
                  </div>
                </li>
              ))}
            </ul>
          </aside>
        )}
      </div>
    </div>
  )
}

function EntityLink({
  id,
  label,
  current,
  onSelect,
}: {
  id: string
  label: string
  current: string
  onSelect: (id: string) => void
}) {
  if (id === current) return <span className="font-medium">{label}</span>
  return (
    <button onClick={() => onSelect(id)} className="font-medium underline decoration-dotted underline-offset-2 hover:text-blue-600 dark:hover:text-blue-300">
      {label}
    </button>
  )
}
