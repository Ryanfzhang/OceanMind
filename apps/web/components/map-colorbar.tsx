import {
  formatMapColorbarValue,
  mapColorbarGradient,
  resolveMapColorScale,
} from "../lib/map-field-preview";
import type { MapColorScale, MapFieldData } from "../lib/types";

type MapColorbarProps = {
  field: MapFieldData | null | undefined;
  compact?: boolean;
  floating?: boolean;
};

export function MapColorbar({ field, compact = false, floating = false }: MapColorbarProps) {
  const colorScale = resolveMapColorScale(field);
  const regionalScales = Array.isArray(field?.regionalColorScales)
    ? field.regionalColorScales.filter((scale) => Number.isFinite(scale.min) && Number.isFinite(scale.max))
    : [];
  const scales = regionalScales.length > 0 ? regionalScales : colorScale ? [colorScale] : [];
  if (scales.length === 0) {
    return null;
  }

  return (
    <div className={`map-colorbar ${compact ? "is-compact" : ""} ${floating ? "is-floating" : ""}`}>
      {scales.map((scale, index) => (
        <ColorbarScale
          key={`${scale.label ?? "scale"}-${index}`}
          scale={scale}
          fallbackLabel={field?.label || field?.variable || "Value"}
          fallbackUnits={field?.units || ""}
        />
      ))}
    </div>
  );
}

function ColorbarScale({
  scale,
  fallbackLabel,
  fallbackUnits,
}: {
  scale: MapColorScale;
  fallbackLabel: string;
  fallbackUnits: string;
}) {
  const label = scale.label || fallbackLabel;
  const units = scale.units || fallbackUnits;
  const isContourScale = String(scale.renderMode ?? "").trim().toLowerCase() === "contours";
  const gradient = `linear-gradient(90deg, ${mapColorbarGradient(13, scale.colormap)})`;

  return (
    <div className="map-colorbar-region">
      <div className="map-colorbar-header">
        <span className="map-colorbar-title">{label}</span>
        {units ? <span className="map-colorbar-units" title={units}>{units}</span> : null}
      </div>
      {isContourScale ? (
        <div className="map-colorbar-contour" aria-hidden="true">
          <span style={{ background: gradient }} />
        </div>
      ) : (
        <div className="map-colorbar-ramp" style={{ background: gradient }} aria-hidden="true" />
      )}
      <div className="map-colorbar-scale">
        <span>{formatMapColorbarValue(scale.min)}</span>
        <span>{formatMapColorbarValue(scale.max)}</span>
      </div>
    </div>
  );
}
