// Utilitários de exportação/compartilhamento do gráfico.
// Sem dependências externas: o gráfico (SVG do recharts) é rasterizado em um
// <canvas> e a partir dele geramos PNG, JPEG e um PDF montado "à mão".

export interface ChartRow {
  name: string;
  value: number;
  unit?: string;
  tooltipLabel?: string;
}

/** Dispara o download de um Blob no navegador. */
export function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  // Revoga depois de um tick para garantir que o download iniciou.
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/**
 * Rasteriza o SVG do gráfico dentro de `container` em um canvas com fundo
 * branco. `scale` aumenta a resolução (2 = retina).
 */
export function chartToCanvas(container: HTMLElement, scale = 2): Promise<HTMLCanvasElement> {
  return new Promise((resolve, reject) => {
    const svg = container.querySelector('svg');
    if (!svg) {
      reject(new Error('Gráfico não encontrado.'));
      return;
    }

    // Clona o SVG e garante atributos de tamanho/namespace explícitos.
    const rect = svg.getBoundingClientRect();
    const width = Math.ceil(rect.width) || 600;
    const height = Math.ceil(rect.height) || 200;

    const clone = svg.cloneNode(true) as SVGSVGElement;
    clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
    clone.setAttribute('width', String(width));
    clone.setAttribute('height', String(height));

    const svgString = new XMLSerializer().serializeToString(clone);
    const svgBlob = new Blob([svgString], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(svgBlob);

    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = width * scale;
      canvas.height = height * scale;
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        URL.revokeObjectURL(url);
        reject(new Error('Canvas indisponível.'));
        return;
      }
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      URL.revokeObjectURL(url);
      resolve(canvas);
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error('Falha ao renderizar o gráfico.'));
    };
    img.src = url;
  });
}

/** Converte um canvas em Blob PNG. */
export function canvasToPngBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error('Falha ao gerar PNG.'));
    }, 'image/png');
  });
}

/** Decodifica uma string base64 em bytes. */
function base64ToBytes(b64: string): Uint8Array {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}

/**
 * Monta um PDF (1.3) de uma página contendo o JPEG do gráfico e um título.
 * O JPEG é embutido diretamente como XObject com filtro DCTDecode.
 */
export function buildChartPdf(canvas: HTMLCanvasElement, title: string): Blob {
  const jpegDataUrl = canvas.toDataURL('image/jpeg', 0.92);
  const jpeg = base64ToBytes(jpegDataUrl.split(',')[1]);

  const imgW = canvas.width;
  const imgH = canvas.height;

  // Página A4 retrato (595 x 842 pt) com margem.
  const pageW = 595;
  const margin = 36;
  const maxW = pageW - margin * 2;
  const drawW = Math.min(maxW, imgW);
  const drawH = (imgH / imgW) * drawW;
  const titleGap = 28;
  const pageH = Math.ceil(drawH + margin * 2 + titleGap);
  const imgX = margin;
  const imgY = margin;
  const titleY = imgY + drawH + 10;

  // Sanitiza o título para WinAnsi básico (sem acentos/parênteses problemáticos).
  const safeTitle = title
    .replace(/[()\\]/g, '')
    .replace(/[–—]/g, '-')
    .replace(/[áàâã]/gi, 'a')
    .replace(/[éê]/gi, 'e')
    .replace(/í/gi, 'i')
    .replace(/[óôõ]/gi, 'o')
    .replace(/[úû]/gi, 'u')
    .replace(/ç/gi, 'c');

  const content =
    `q\n${drawW.toFixed(2)} 0 0 ${drawH.toFixed(2)} ${imgX} ${imgY} cm\n/Im0 Do\nQ\n` +
    `BT /F1 14 Tf 0.1 0.23 0.36 rg ${margin} ${titleY.toFixed(2)} Td (${safeTitle}) Tj ET\n`;

  const enc = (s: string) => {
    const bytes = new Uint8Array(s.length);
    for (let i = 0; i < s.length; i++) bytes[i] = s.charCodeAt(i) & 0xff;
    return bytes;
  };

  const chunks: Uint8Array[] = [];
  let offset = 0;
  const offsets: number[] = [];
  const push = (part: Uint8Array) => {
    chunks.push(part);
    offset += part.length;
  };
  const pushStr = (s: string) => push(enc(s));
  const startObj = () => offsets.push(offset);

  pushStr('%PDF-1.3\n');

  startObj();
  pushStr('1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n');

  startObj();
  pushStr('2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n');

  startObj();
  pushStr(
    `3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 ${pageW} ${pageH}] ` +
      `/Resources << /XObject << /Im0 4 0 R >> /Font << /F1 6 0 R >> >> /Contents 5 0 R >>\nendobj\n`,
  );

  startObj();
  pushStr(
    `4 0 obj\n<< /Type /XObject /Subtype /Image /Width ${imgW} /Height ${imgH} ` +
      `/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length ${jpeg.length} >>\nstream\n`,
  );
  push(jpeg);
  pushStr('\nendstream\nendobj\n');

  startObj();
  const contentBytes = enc(content);
  pushStr(`5 0 obj\n<< /Length ${contentBytes.length} >>\nstream\n`);
  push(contentBytes);
  pushStr('\nendstream\nendobj\n');

  startObj();
  pushStr('6 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n');

  const xrefStart = offset;
  let xref = `xref\n0 ${offsets.length + 1}\n0000000000 65535 f \n`;
  for (const off of offsets) {
    xref += `${String(off).padStart(10, '0')} 00000 n \n`;
  }
  xref += `trailer\n<< /Size ${offsets.length + 1} /Root 1 0 R >>\nstartxref\n${xrefStart}\n%%EOF`;
  pushStr(xref);

  return new Blob(chunks as BlobPart[], { type: 'application/pdf' });
}

/** Gera um CSV (separado por vírgula) a partir das linhas do gráfico. */
export function buildCsv(rows: ChartRow[]): Blob {
  const escape = (v: string) => `"${v.replace(/"/g, '""')}"`;
  const header = ['Setor', 'Valor', 'Unidade', 'Métrica'];
  const lines = [header.map(escape).join(',')];
  for (const r of rows) {
    lines.push(
      [escape(r.name), String(r.value), escape(r.unit ?? ''), escape(r.tooltipLabel ?? '')].join(','),
    );
  }
  // BOM para o Excel reconhecer UTF-8 (acentos).
  return new Blob(['﻿' + lines.join('\r\n')], { type: 'text/csv;charset=utf-8' });
}
