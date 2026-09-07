//! Chat Markdown renderer: pulldown-cmark events -> egui widgets.
//!
//! Covers what LLMs actually emit: headings, bold/italic/strikethrough,
//! inline code, fenced code blocks, bullet/numbered/task lists (nested),
//! blockquotes, links, tables (GFM), horizontal rules. Raw HTML is shown
//! as plain text (safer than rendering it).
//!
//! [`parse`] is pure and unit-tested; [`show`] renders one message.
//! Re-parsing every frame is cheap at chat sizes, which also makes
//! streaming output render live.
#![forbid(unsafe_code)]

use pulldown_cmark::{Alignment, CodeBlockKind, Event, Options, Parser, Tag, TagEnd};

#[derive(Debug, Clone, Default)]
pub struct Span {
    pub text: String,
    pub strong: bool,
    pub em: bool,
    pub strike: bool,
    pub code: bool,
    pub link: Option<String>,
}

impl Span {
    fn plain(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            ..Default::default()
        }
    }

    fn is_empty(&self) -> bool {
        self.text.is_empty()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellAlign {
    Left,
    Center,
    Right,
}

#[derive(Debug, Clone)]
pub enum Block {
    Heading {
        level: u8,
        spans: Vec<Span>,
    },
    Para {
        spans: Vec<Span>,
        indent: usize,
        quote: bool,
    },
    Code {
        lang: String,
        code: String,
    },
    ListItem {
        ordered: bool,
        index: u64,
        checked: Option<bool>,
        spans: Vec<Span>,
        indent: usize,
        /// Document order: nested items close before their parents, so a
        /// final pass re-sorts consecutive runs by this key.
        seq: u64,
    },
    Table {
        headers: Vec<Vec<Span>>,
        aligns: Vec<CellAlign>,
        rows: Vec<Vec<Vec<Span>>>,
    },
    Rule,
}

struct Inline {
    strong: u32,
    em: u32,
    strike: u32,
    code: bool,
    link: Vec<String>,
}

impl Inline {
    fn span(&self, text: &str) -> Option<Span> {
        if text.is_empty() {
            return None;
        }
        Some(Span {
            text: text.to_string(),
            strong: self.strong > 0,
            em: self.em > 0,
            strike: self.strike > 0,
            code: self.code,
            link: self.link.last().cloned(),
        })
    }
}

struct ListCtx {
    ordered: bool,
    next: u64,
}

/// One open list item; stacked so nested lists keep their parent's text.
struct ItemBuilder {
    spans: Vec<Span>,
    checked: Option<bool>,
    indent: usize,
    seq: u64,
}

struct TableBuilder {
    aligns: Vec<CellAlign>,
    headers: Vec<Vec<Span>>,
    rows: Vec<Vec<Vec<Span>>>,
    row: Vec<Vec<Span>>,
}

fn md_options() -> Options {
    Options::ENABLE_TABLES
        | Options::ENABLE_STRIKETHROUGH
        | Options::ENABLE_TASKLISTS
        | Options::ENABLE_FOOTNOTES
}

/// Parse Markdown into render blocks.
pub fn parse(text: &str) -> Vec<Block> {
    let mut blocks: Vec<Block> = Vec::new();
    let mut inline = Inline {
        strong: 0,
        em: 0,
        strike: 0,
        code: false,
        link: Vec::new(),
    };
    let mut spans: Vec<Span> = Vec::new();
    let mut quote_depth: usize = 0;
    let mut lists: Vec<ListCtx> = Vec::new();
    let mut items: Vec<ItemBuilder> = Vec::new();
    let mut item_seq: u64 = 0;
    let mut code_lang = String::new();
    let mut code_buf = String::new();
    let mut in_code = false;
    let mut table: Option<TableBuilder> = None;
    let mut skip_depth: usize = 0; // footnote definitions, metadata: dropped

    macro_rules! push_span {
        ($t:expr) => {
            if let Some(s) = inline.span($t) {
                spans.push(s);
            }
        };
    }

    for event in Parser::new_ext(text, md_options()) {
        if skip_depth > 0 {
            match &event {
                Event::Start(_) => skip_depth += 1,
                Event::End(_) => skip_depth -= 1,
                _ => {}
            }
            continue;
        }
        match event {
            Event::Start(tag) => match tag {
                Tag::Paragraph => {}
                Tag::Heading { level, .. } => {
                    spans.clear();
                    blocks.push(Block::Heading {
                        level: level as u8,
                        spans: Vec::new(),
                    });
                }
                Tag::BlockQuote(_) => quote_depth += 1,
                Tag::CodeBlock(kind) => {
                    in_code = true;
                    code_buf.clear();
                    code_lang = match kind {
                        CodeBlockKind::Fenced(lang) => {
                            lang.split_whitespace().next().unwrap_or("").to_string()
                        }
                        CodeBlockKind::Indented => String::new(),
                    };
                }
                Tag::List(first) => {
                    lists.push(ListCtx {
                        ordered: first.is_some(),
                        next: first.unwrap_or(1),
                    });
                }
                Tag::Item => {
                    items.push(ItemBuilder {
                        spans: Vec::new(),
                        checked: None,
                        indent: lists.len().saturating_sub(1),
                        seq: item_seq,
                    });
                    item_seq += 1;
                }
                Tag::Table(aligns) => {
                    table = Some(TableBuilder {
                        aligns: aligns
                            .iter()
                            .map(|a| match a {
                                Alignment::Left | Alignment::None => CellAlign::Left,
                                Alignment::Center => CellAlign::Center,
                                Alignment::Right => CellAlign::Right,
                            })
                            .collect(),
                        headers: Vec::new(),
                        rows: Vec::new(),
                        row: Vec::new(),
                    });
                }
                Tag::TableHead => {}
                Tag::TableRow => {
                    if let Some(t) = table.as_mut() {
                        t.row.clear();
                    }
                }
                Tag::TableCell => {
                    spans.clear();
                }
                Tag::Emphasis => inline.em += 1,
                Tag::Strong => inline.strong += 1,
                Tag::Strikethrough => inline.strike += 1,
                Tag::Link { dest_url, .. } => inline.link.push(dest_url.into_string()),
                Tag::Image { dest_url, .. } => {
                    inline.link.push(dest_url.into_string());
                }
                Tag::FootnoteDefinition(_) | Tag::MetadataBlock(_) => skip_depth = 1,
                _ => {}
            },
            Event::End(end) => match end {
                TagEnd::Paragraph => {
                    if !spans.iter().all(|s| s.is_empty()) {
                        let quote = quote_depth > 0;
                        let indent = lists.len();
                        blocks.push(Block::Para {
                            spans: std::mem::take(&mut spans),
                            indent,
                            quote,
                        });
                    } else {
                        spans.clear();
                    }
                }
                TagEnd::Heading(_) => {
                    if let Some(Block::Heading { spans: slot, .. }) = blocks.last_mut() {
                        *slot = std::mem::take(&mut spans);
                    }
                }
                TagEnd::BlockQuote(_) => quote_depth = quote_depth.saturating_sub(1),
                TagEnd::CodeBlock => {
                    in_code = false;
                    blocks.push(Block::Code {
                        lang: std::mem::take(&mut code_lang),
                        code: std::mem::take(&mut code_buf),
                    });
                }
                TagEnd::List(_) => {
                    lists.pop();
                }
                TagEnd::Item => {
                    if let Some(built) = items.pop() {
                        if let Some(ctx) = lists.last_mut() {
                            let index = ctx.next;
                            if ctx.ordered {
                                ctx.next += 1;
                            }
                            blocks.push(Block::ListItem {
                                ordered: ctx.ordered,
                                index,
                                checked: built.checked,
                                spans: built.spans,
                                indent: built.indent,
                                seq: built.seq,
                            });
                        } else {
                            blocks.push(Block::Para {
                                spans: built.spans,
                                indent: 0,
                                quote: false,
                            });
                        }
                    }
                    spans.clear();
                }
                TagEnd::Table => {
                    if let Some(t) = table.take()
                        && (!t.headers.is_empty() || !t.rows.is_empty())
                    {
                        blocks.push(Block::Table {
                            headers: t.headers,
                            aligns: t.aligns,
                            rows: t.rows,
                        });
                    }
                }
                TagEnd::TableHead => {
                    if let Some(t) = table.as_mut() {
                        // The head holds bare cells (no TableRow wrapper).
                        t.headers = std::mem::take(&mut t.row);
                    }
                }
                TagEnd::TableRow => {
                    if let Some(t) = table.as_mut() {
                        let row = std::mem::take(&mut t.row);
                        if !row.iter().all(|c| c.iter().all(|s| s.is_empty())) {
                            t.rows.push(row);
                        }
                    }
                }
                TagEnd::TableCell => {
                    if let Some(t) = table.as_mut() {
                        t.row.push(std::mem::take(&mut spans));
                    }
                }
                TagEnd::Emphasis => inline.em = inline.em.saturating_sub(1),
                TagEnd::Strong => inline.strong = inline.strong.saturating_sub(1),
                TagEnd::Strikethrough => inline.strike = inline.strike.saturating_sub(1),
                TagEnd::Link | TagEnd::Image => {
                    inline.link.pop();
                }
                _ => {}
            },
            Event::Text(t) => {
                if in_code {
                    code_buf.push_str(&t);
                } else if table.is_none() {
                    if let Some(top) = items.last_mut() {
                        if let Some(s) = inline.span(&t) {
                            top.spans.push(s);
                        }
                    } else {
                        push_span!(&t);
                    }
                } else {
                    push_span!(&t);
                }
            }
            Event::Code(t) => {
                let span = Span {
                    text: t.into_string(),
                    code: true,
                    link: inline.link.last().cloned(),
                    ..Default::default()
                };
                if table.is_none() {
                    if let Some(top) = items.last_mut() {
                        top.spans.push(span);
                    } else {
                        spans.push(span);
                    }
                } else {
                    spans.push(span);
                }
            }
            Event::Html(t) | Event::InlineHtml(t) => {
                push_span!(t.as_ref());
            }
            Event::SoftBreak => {
                // Chat-friendly: single newlines break the line.
                if in_code {
                    code_buf.push('\n');
                } else if table.is_none() {
                    if let Some(top) = items.last_mut() {
                        top.spans.push(Span::plain("\n"));
                    } else {
                        push_span!("\n");
                    }
                } else {
                    push_span!("\n");
                }
            }
            Event::HardBreak => {
                if !in_code {
                    push_span!("\n");
                }
            }
            Event::Rule => {
                if !in_code {
                    flush_para(&mut blocks, &mut spans, lists.len(), quote_depth > 0);
                    blocks.push(Block::Rule);
                }
            }
            Event::TaskListMarker(checked) => {
                if let Some(top) = items.last_mut() {
                    top.checked = Some(checked);
                }
            }
            Event::FootnoteReference(name) => {
                push_span!(&format!("[^{}]", name.as_ref()));
            }
            _ => {}
        }
    }
    // Unclosed paragraphs AND list items at EOF (streaming text): keep
    // what we have instead of dropping it.
    while !items.is_empty() {
        let built = items.remove(0);
        if let Some(ctx) = lists.last_mut() {
            let index = ctx.next;
            if ctx.ordered {
                ctx.next += 1;
            }
            blocks.push(Block::ListItem {
                ordered: ctx.ordered,
                index,
                checked: built.checked,
                spans: built.spans,
                indent: built.indent,
                seq: built.seq,
            });
        } else {
            blocks.push(Block::Para {
                spans: built.spans,
                indent: 0,
                quote: false,
            });
        }
    }
    if !spans.iter().all(|s| s.is_empty()) {
        let quote = quote_depth > 0;
        let indent = lists.len();
        blocks.push(Block::Para {
            spans,
            indent,
            quote,
        });
    }
    // Nested items close before their parents; restore document order
    // within each consecutive run of list items (stable sort).
    let mut i = 0;
    while i < blocks.len() {
        if matches!(blocks[i], Block::ListItem { .. }) {
            let mut j = i;
            while j < blocks.len() && matches!(blocks[j], Block::ListItem { .. }) {
                j += 1;
            }
            blocks[i..j].sort_by_key(|b| match b {
                Block::ListItem { seq, .. } => *seq,
                _ => u64::MAX,
            });
            i = j;
        } else {
            i += 1;
        }
    }
    blocks
}

fn flush_para(blocks: &mut Vec<Block>, spans: &mut Vec<Span>, indent: usize, quote: bool) {
    if !spans.iter().all(|s| s.is_empty()) {
        blocks.push(Block::Para {
            spans: std::mem::take(spans),
            indent,
            quote,
        });
    } else {
        spans.clear();
    }
}

fn level_size(level: u8) -> f32 {
    match level {
        1 => 22.0,
        2 => 19.0,
        3 => 17.0,
        _ => 15.0,
    }
}

fn span_job(spans: &[Span], base_size: f32) -> egui::text::LayoutJob {
    use egui::text::{LayoutJob, TextFormat};
    let mut job = LayoutJob::default();
    for s in spans {
        let mut fmt = TextFormat {
            font_id: if s.code {
                egui::FontId::monospace(base_size - 1.0)
            } else {
                egui::FontId::proportional(base_size)
            },
            color: if s.link.is_some() {
                egui::Color32::LIGHT_BLUE
            } else {
                egui::Color32::PLACEHOLDER
            },
            italics: s.em,
            underline: if s.link.is_some() {
                egui::Stroke::new(1.0, egui::Color32::LIGHT_BLUE)
            } else {
                egui::Stroke::NONE
            },
            strikethrough: if s.strike {
                egui::Stroke::new(1.0, egui::Color32::GRAY)
            } else {
                egui::Stroke::NONE
            },
            background: if s.code {
                egui::Color32::from_rgba_unmultiplied(255, 255, 255, 18)
            } else {
                egui::Color32::TRANSPARENT
            },
            ..Default::default()
        };
        if s.strong {
            fmt.color = egui::Color32::WHITE;
        }
        job.append(&s.text, 0.0, fmt);
    }
    job
}

/// Render one chat message. `salt` scopes widget ids (tables,
/// collapsibles) so identical messages don't share UI state.
pub fn show(ui: &mut egui::Ui, salt: &str, text: &str) {
    ui.push_id(salt, |ui| {
        let mut table_seq = 0usize;
        for block in parse(text) {
            match block {
                Block::Heading { level, spans } => {
                    ui.add_space(4.0);
                    let mut job = span_job(&spans, level_size(level));
                    for section in job.sections.iter_mut() {
                        section.format.font_id.size = level_size(level);
                        section.format.color = egui::Color32::WHITE;
                    }
                    ui.label(job);
                }
                Block::Para {
                    spans,
                    indent,
                    quote,
                } => {
                    let job = span_job(&spans, 14.0);
                    if indent > 0 || quote {
                        ui.horizontal(|ui| {
                            ui.add_space((indent as f32) * 8.0);
                            if quote {
                                ui.label(egui::RichText::new("│").weak().small());
                            }
                            ui.label(job);
                        });
                    } else {
                        ui.label(job);
                    }
                }
                Block::Code { lang, code } => {
                    egui::Frame::group(ui.style()).show(ui, |ui| {
                        if !lang.is_empty() {
                            ui.label(egui::RichText::new(&lang).small().weak().monospace());
                        }
                        ui.add(
                            egui::Label::new(egui::RichText::new(code).monospace())
                                .selectable(true),
                        );
                    });
                }
                Block::ListItem {
                    ordered,
                    index,
                    checked,
                    spans,
                    indent,
                    ..
                } => {
                    let marker = match checked {
                        Some(true) => "[x] ".to_string(),
                        Some(false) => "[ ] ".to_string(),
                        None if ordered => format!("{index}. "),
                        None => "• ".to_string(),
                    };
                    let job = span_job(&spans, 14.0);
                    ui.horizontal(|ui| {
                        ui.add_space((indent as f32) * 18.0);
                        ui.label(egui::RichText::new(marker).weak());
                        ui.label(job);
                    });
                }
                Block::Table {
                    headers,
                    aligns,
                    rows,
                } => {
                    table_seq += 1;
                    egui::Grid::new(format!("md-table-{table_seq}"))
                        .striped(true)
                        .spacing([14.0, 4.0])
                        .show(ui, |ui| {
                            for (i, head) in headers.iter().enumerate() {
                                cell(ui, &span_job(head, 13.5), aligns.get(i), true);
                            }
                            ui.end_row();
                            for row in &rows {
                                for (i, cell_spans) in row.iter().enumerate() {
                                    cell(ui, &span_job(cell_spans, 13.5), aligns.get(i), false);
                                }
                                ui.end_row();
                            }
                        });
                }
                Block::Rule => {
                    ui.separator();
                }
            }
        }
    });
}

fn cell(ui: &mut egui::Ui, job: &egui::text::LayoutJob, align: Option<&CellAlign>, header: bool) {
    let mut job = job.clone();
    if header {
        for section in job.sections.iter_mut() {
            section.format.color = egui::Color32::WHITE;
        }
    }
    match align {
        Some(CellAlign::Right) => {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(job);
            });
        }
        Some(CellAlign::Center) => {
            ui.with_layout(egui::Layout::top_down(egui::Align::Center), |ui| {
                ui.label(job)
            });
        }
        _ => {
            ui.label(job);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texts(blocks: &[Block]) -> Vec<String> {
        blocks
            .iter()
            .filter_map(|b| match b {
                Block::Para { spans, .. } => Some(
                    spans
                        .iter()
                        .map(|s| s.text.as_str())
                        .collect::<Vec<_>>()
                        .join(""),
                ),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn headings_bold_code_spans() {
        let blocks = parse("# Title\n\nHello **bold** and `code`.\n");
        assert!(matches!(&blocks[0], Block::Heading { level: 1, .. }));
        assert_eq!(texts(&blocks), vec!["Hello bold and code.".to_string()]);
        if let Block::Para { spans, .. } = &blocks[1] {
            assert!(spans.iter().any(|s| s.strong && s.text == "bold"));
            assert!(spans.iter().any(|s| s.code && s.text == "code"));
        } else {
            panic!("expected para");
        }
    }

    #[test]
    fn tables_parse_with_alignment() {
        let md = "| A | B |\n|---|---:|\n| 1 | 2 |\n| 3 | 4 |\n";
        let blocks = parse(md);
        let table = blocks.iter().find_map(|b| match b {
            Block::Table {
                headers,
                aligns,
                rows,
            } => Some((headers, aligns, rows)),
            _ => None,
        });
        let (headers, aligns, rows) = table.expect("table block");
        assert_eq!(headers.len(), 2);
        assert_eq!(aligns, &vec![CellAlign::Left, CellAlign::Right]);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0][0][0].text, "1");
    }

    #[test]
    fn nested_lists_and_tasks() {
        let md = "- a\n  - b\n1. one\n- [x] done\n- [ ] todo\n";
        let blocks = parse(md);
        let items: Vec<_> = blocks
            .iter()
            .filter_map(|b| match b {
                Block::ListItem {
                    ordered,
                    index,
                    checked,
                    indent,
                    ..
                } => Some((*ordered, *index, *checked, *indent)),
                _ => None,
            })
            .collect();
        assert_eq!(
            items,
            vec![
                (false, 1, None, 0),
                (false, 1, None, 1),
                (true, 1, None, 0),
                (false, 1, Some(true), 0),
                (false, 1, Some(false), 0),
            ]
        );
        // task markers land on the right items
        let checks: Vec<_> = blocks
            .iter()
            .filter_map(|b| match b {
                Block::ListItem { checked, .. } => Some(*checked),
                _ => None,
            })
            .collect();
        assert_eq!(checks, vec![None, None, None, Some(true), Some(false)]);
    }

    #[test]
    fn fenced_code_keeps_lang_and_text() {
        let blocks = parse("```rust\nfn main() {}\n```\n");
        assert!(matches!(
            &blocks[0],
            Block::Code { lang, code }
            if lang == "rust" && code.trim() == "fn main() {}"
        ));
    }

    #[test]
    fn quotes_and_rules() {
        let blocks = parse("> wisdom\n\n---\n");
        assert!(matches!(&blocks[0], Block::Para { quote: true, .. }));
        assert!(matches!(&blocks[1], Block::Rule));
    }

    #[test]
    fn streaming_prefix_stays_readable() {
        // A half-written table or code fence must not panic or vanish.
        let blocks = parse("Here is **bold");
        assert!(!texts(&blocks).is_empty() || !blocks.is_empty());
        let blocks = parse("| A | B |\n|---|\n| 1 |");
        let _ = blocks;
        let blocks = parse("```py\nprint(1)");
        assert!(matches!(&blocks[0], Block::Code { .. }));
    }
}
