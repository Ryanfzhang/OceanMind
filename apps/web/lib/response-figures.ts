import type { ResultCardSummary } from "./types";

export type ResponseFigure = { card: ResultCardSummary; caption: string };

function hasVisualContent(card: ResultCardSummary) {
  const data = card.workspaceData;
  return card.type === "image_png" || Boolean(
    data?.mapField || data?.mapFieldFrames?.length || data?.eventOverlays?.length ||
    data?.resultSeries?.length || data?.profileSeries?.length ||
    data?.hovmollerRows?.length || data?.sectionRows?.length ||
    data?.eofModes?.length || data?.compositeFields?.length ||
    data?.histogramBins?.length || data?.tsDiagramPoints?.length,
  );
}

export function selectResponseFigures(text: string, cards: ResultCardSummary[]) {
  const cardsById = new Map(cards.map((card) => [card.id, card]));
  const figures: ResponseFigure[] = [];
  const prose = text.replace(/!\[([^\]]+)\]\(#figure-([a-zA-Z0-9_]+)\)/g, (full, caption: string, id: string) => {
    const card = cardsById.get(id);
    if (!card || !hasVisualContent(card)) return caption;
    if (figures.some((figure) => figure.card.id === id)) return "";
    figures.push({ card, caption });
    return "";
  });
  return { prose, figures };
}
