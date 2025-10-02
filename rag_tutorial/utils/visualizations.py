"""
Visualization utilities for RAG tutorial
Generates flowcharts and diagrams to explain RAG concepts
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


class RAGVisualizer:
    """Create visual diagrams explaining RAG concepts"""

    def __init__(self, output_dir='diagrams'):
        self.output_dir = output_dir
        self.colors = {
            'primary': '#3498db',
            'secondary': '#2ecc71',
            'tertiary': '#e74c3c',
            'gray': '#95a5a6',
            'light': '#ecf0f1',
            'dark': '#34495e'
        }

    def draw_rag_overview(self, save=True):
        """Draw high-level RAG system overview"""
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.axis('off')

        # Title
        ax.text(5, 11.5, 'RAG System Overview',
                ha='center', fontsize=20, fontweight='bold')

        # User Question
        self._draw_box(ax, 4, 10, 2, 0.8, 'User Question\n"What is the remote work policy?"',
                      self.colors['tertiary'])

        # Arrow down
        self._draw_arrow(ax, 5, 10, 5, 8.5)

        # RAG System (main box)
        system_box = FancyBboxPatch((1, 2), 8, 6,
                                    boxstyle="round,pad=0.1",
                                    edgecolor=self.colors['primary'],
                                    facecolor=self.colors['light'],
                                    linewidth=3)
        ax.add_patch(system_box)
        ax.text(5, 7.5, 'RAG SYSTEM', ha='center', fontsize=14, fontweight='bold',
               color=self.colors['primary'])

        # Step 1: Retrieval
        self._draw_box(ax, 1.5, 6, 2.5, 0.8, '1. RETRIEVAL\nSearch vector DB\nfor relevant docs',
                      self.colors['primary'])

        # Arrow
        self._draw_arrow(ax, 4.25, 6.4, 5.5, 6.4)

        # Step 2: Augmentation
        self._draw_box(ax, 5.5, 6, 2.5, 0.8, '2. AUGMENTATION\nCombine query +\nrelevant chunks',
                      self.colors['secondary'])

        # Arrow down
        self._draw_arrow(ax, 5, 6, 5, 4.5)

        # Step 3: Generation
        self._draw_box(ax, 3.5, 3.5, 3, 0.8, '3. GENERATION\nLLM generates answer\nbased on context',
                      self.colors['tertiary'])

        # Components on sides
        # Left: Vector DB
        self._draw_box(ax, 1.5, 4, 2, 0.6, 'Vector Database\n(Document Index)',
                      self.colors['gray'], alpha=0.7)
        self._draw_arrow(ax, 2.5, 4.6, 2.5, 6, style='dashed')

        # Right: LLM
        self._draw_box(ax, 6.5, 4, 2, 0.6, 'Large Language\nModel (GPT/Claude)',
                      self.colors['gray'], alpha=0.7)
        self._draw_arrow(ax, 7.5, 4.6, 6.5, 3.9, style='dashed')

        # Final Answer
        self._draw_arrow(ax, 5, 3.5, 5, 2)
        self._draw_box(ax, 3.5, 0.8, 3, 0.8,
                      'Answer with Sources\n"Employees can work remotely\nup to 3 days/week..."',
                      self.colors['secondary'])

        plt.tight_layout()

        if save:
            plt.savefig(f'{self.output_dir}/rag_overview.png', dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {self.output_dir}/rag_overview.png")

        return fig

    def draw_indexing_pipeline(self, save=True):
        """Draw the document indexing pipeline"""
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 8)
        ax.axis('off')

        # Title
        ax.text(8, 7.5, 'Document Indexing Pipeline (One-Time Setup)',
                ha='center', fontsize=18, fontweight='bold')

        y_pos = 5

        # Step 1: Documents
        self._draw_box(ax, 0.5, y_pos, 2.5, 1.2,
                      'Documents\n\n📄 PDFs\n📝 Text files\n📊 Reports',
                      self.colors['gray'])

        # Arrow
        self._draw_arrow(ax, 3.2, y_pos + 0.6, 4, y_pos + 0.6)

        # Step 2: Text Extraction
        self._draw_box(ax, 4, y_pos, 2.5, 1.2,
                      'Text Extraction\n\nParse & Clean\nRemove formatting',
                      self.colors['primary'])

        # Arrow
        self._draw_arrow(ax, 6.7, y_pos + 0.6, 7.5, y_pos + 0.6)

        # Step 3: Chunking
        self._draw_box(ax, 7.5, y_pos, 2.5, 1.2,
                      'Chunking\n\nSplit into\n200-500 word\nchunks',
                      self.colors['secondary'])

        # Arrow
        self._draw_arrow(ax, 10.2, y_pos + 0.6, 11, y_pos + 0.6)

        # Step 4: Embeddings
        self._draw_box(ax, 11, y_pos, 2.5, 1.2,
                      'Embeddings\n\nConvert to\nvectors using\nneural model',
                      self.colors['tertiary'])

        # Arrow
        self._draw_arrow(ax, 13.7, y_pos + 0.6, 14.5, y_pos + 0.6)

        # Step 5: Vector DB
        self._draw_box(ax, 14.5, y_pos, 1.3, 1.2,
                      'Store in\nVector\nDB',
                      self.colors['primary'])

        # Examples below
        examples_y = 2.5

        # Example chunk
        ax.text(5.25, examples_y + 0.5, 'Example Chunk:', fontsize=10, fontweight='bold')
        chunk_text = '"Employees may work\\nremotely up to 3 days\\nper week..."'
        self._draw_box(ax, 4, examples_y - 0.8, 2.5, 0.8, chunk_text,
                      self.colors['light'], textsize=8)

        # Example embedding
        ax.text(8.75, examples_y + 0.5, 'Becomes Vector:', fontsize=10, fontweight='bold')
        embedding_text = '[0.23, -0.45, 0.67,\\n0.12, -0.89, ...]\\n(384 dimensions)'
        self._draw_box(ax, 7.5, examples_y - 0.8, 2.5, 0.8, embedding_text,
                      self.colors['light'], textsize=8)

        # Example search
        ax.text(12.25, examples_y + 0.5, 'Fast Search:', fontsize=10, fontweight='bold')
        search_text = 'Similarity search\\nin milliseconds\\nacross millions\\nof vectors'
        self._draw_box(ax, 11, examples_y - 0.8, 2.5, 0.8, search_text,
                      self.colors['light'], textsize=8)

        plt.tight_layout()

        if save:
            plt.savefig(f'{self.output_dir}/indexing_pipeline.png', dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {self.output_dir}/indexing_pipeline.png")

        return fig

    def draw_query_pipeline(self, save=True):
        """Draw the query/inference pipeline"""
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 10)
        ax.axis('off')

        # Title
        ax.text(7, 9.5, 'Query Pipeline (Every Question)',
                ha='center', fontsize=18, fontweight='bold')

        # User question
        self._draw_box(ax, 5.5, 8.5, 3, 0.7,
                      'User: "What is the PTO policy?"',
                      self.colors['tertiary'])
        self._draw_arrow(ax, 7, 8.5, 7, 7.5)

        # Step 1: Embed query
        self._draw_box(ax, 5, 6.8, 4, 0.7,
                      'Step 1: Embed Query\nconvert question → vector',
                      self.colors['primary'])
        ax.text(9.5, 7.15, '[0.31, -0.22, ...]', fontsize=8, style='italic')
        self._draw_arrow(ax, 7, 6.8, 7, 6)

        # Step 2: Search vector DB
        self._draw_box(ax, 5, 5.3, 4, 0.7,
                      'Step 2: Search Vector DB\nfind top-k similar chunks',
                      self.colors['secondary'])

        # Vector DB on side
        self._draw_box(ax, 0.5, 4.5, 3, 1.5,
                      'Vector Database\n\n1M+ document\nchunks indexed',
                      self.colors['gray'], alpha=0.7)
        self._draw_arrow(ax, 3.5, 5.6, 5, 5.6, style='dashed')

        self._draw_arrow(ax, 7, 5.3, 7, 4.5)

        # Retrieved chunks
        self._draw_box(ax, 1, 3.3, 3.5, 0.6,
                      'Chunk 1: PTO policy...\nScore: 0.92',
                      self.colors['light'], textsize=8)
        self._draw_box(ax, 4.75, 3.3, 3.5, 0.6,
                      'Chunk 2: Leave benefits...\nScore: 0.87',
                      self.colors['light'], textsize=8)
        self._draw_box(ax, 8.5, 3.3, 3.5, 0.6,
                      'Chunk 3: Time off rules...\nScore: 0.81',
                      self.colors['light'], textsize=8)

        self._draw_arrow(ax, 7, 3.3, 7, 2.5)

        # Step 3: Build context
        self._draw_box(ax, 4.5, 1.8, 5, 0.7,
                      'Step 3: Build Context\nQuestion + Retrieved Chunks → Prompt',
                      self.colors['primary'])

        # LLM on side
        self._draw_box(ax, 10.5, 1, 3, 1.5,
                      'Large Language\nModel\n\nGPT-4 or\nClaude',
                      self.colors['gray'], alpha=0.7)
        self._draw_arrow(ax, 9.5, 2.15, 10.5, 2.15, style='dashed')

        self._draw_arrow(ax, 7, 1.8, 7, 1)

        # Final answer
        self._draw_box(ax, 4.5, 0.2, 5, 0.7,
                      'Answer: "New employees get 15 PTO days..."\\nSources: [company_policies.txt]',
                      self.colors['secondary'])

        plt.tight_layout()

        if save:
            plt.savefig(f'{self.output_dir}/query_pipeline.png', dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {self.output_dir}/query_pipeline.png")

        return fig

    def draw_chunking_strategies(self, save=True):
        """Visualize different chunking strategies"""
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        # Sample text
        sample_text = """SECTION 1: WORK HOURS

Full-time employees work 40 hours per week. We offer flexible working arrangements.
Remote work is available up to 3 days per week.

SECTION 2: LEAVE POLICIES

Employees receive 15-25 days of PTO annually. Sick leave is separate.
Medical certificates required for 3+ consecutive days."""

        # Strategy 1: Fixed-size
        ax = axes[0]
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 3)
        ax.axis('off')
        ax.text(5, 2.7, 'Fixed-Size Chunking (Every 100 characters)',
                ha='center', fontsize=12, fontweight='bold')

        chunks_fixed = [sample_text[i:i+100] for i in range(0, len(sample_text), 100)]
        for i, chunk in enumerate(chunks_fixed[:5]):
            self._draw_box(ax, 0.5 + i*2, 0.5, 1.8, 2,
                          f'Chunk {i+1}\n{len(chunk)} chars',
                          self.colors['primary'], ax=ax, textsize=8)

        # Strategy 2: Sentence-based
        ax = axes[1]
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 3)
        ax.axis('off')
        ax.text(5, 2.7, 'Sentence-Based Chunking (2 sentences per chunk)',
                ha='center', fontsize=12, fontweight='bold')

        sentences = sample_text.split('. ')
        chunks_sentence = ['. '.join(sentences[i:i+2]) for i in range(0, len(sentences), 2)]
        for i, chunk in enumerate(chunks_sentence[:4]):
            self._draw_box(ax, 0.5 + i*2.4, 0.5, 2.2, 2,
                          f'Chunk {i+1}\n{len(chunk)} chars',
                          self.colors['secondary'], ax=ax, textsize=8)

        # Strategy 3: Semantic
        ax = axes[2]
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 3)
        ax.axis('off')
        ax.text(5, 2.7, 'Semantic Chunking (Split on sections)',
                ha='center', fontsize=12, fontweight='bold')

        sections = sample_text.split('SECTION')
        chunks_semantic = [s.strip() for s in sections if s.strip()]
        for i, chunk in enumerate(chunks_semantic[:3]):
            self._draw_box(ax, 1 + i*3, 0.5, 2.8, 2,
                          f'Chunk {i+1}\n{len(chunk)} chars\n(Full section)',
                          self.colors['tertiary'], ax=ax, textsize=8)

        plt.tight_layout()

        if save:
            plt.savefig(f'{self.output_dir}/chunking_strategies.png', dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {self.output_dir}/chunking_strategies.png")

        return fig

    def draw_embedding_similarity(self, save=True):
        """Visualize how embeddings capture semantic similarity"""
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_xlim(-2, 12)
        ax.set_ylim(-2, 10)
        ax.axis('off')

        # Title
        ax.text(5, 9.5, 'Semantic Similarity with Vector Embeddings',
                ha='center', fontsize=16, fontweight='bold')

        # Query
        query_pos = (5, 7.5)
        self._draw_box(ax, query_pos[0]-1, query_pos[1]-0.3, 2, 0.6,
                      'Query:\n"remote work"',
                      self.colors['tertiary'])

        # Similar documents (close)
        similar_docs = [
            ('Doc 1:\n"work from home\npolicy"', (3, 5), 0.95),
            ('Doc 2:\n"telecommuting\nrules"', (5, 4.5), 0.91),
            ('Doc 3:\n"flexible work\narrangements"', (7, 5), 0.88)
        ]

        for doc_text, pos, score in similar_docs:
            self._draw_box(ax, pos[0]-0.8, pos[1]-0.3, 1.6, 0.6,
                          doc_text, self.colors['secondary'], textsize=8)
            # Draw line showing similarity
            ax.plot([query_pos[0], pos[0]], [query_pos[1], pos[1]],
                   'g-', linewidth=2, alpha=0.6)
            mid_x, mid_y = (query_pos[0] + pos[0])/2, (query_pos[1] + pos[1])/2
            ax.text(mid_x, mid_y, f'{score:.2f}', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

        # Dissimilar documents (far)
        dissimilar_docs = [
            ('Doc 4:\n"coffee machine\nmanual"', (1, 2), 0.12),
            ('Doc 5:\n"parking lot\nreservations"', (9, 2), 0.08)
        ]

        for doc_text, pos, score in dissimilar_docs:
            self._draw_box(ax, pos[0]-0.8, pos[1]-0.3, 1.6, 0.6,
                          doc_text, self.colors['gray'], textsize=8)
            # Draw line showing low similarity
            ax.plot([query_pos[0], pos[0]], [query_pos[1], pos[1]],
                   'r--', linewidth=1, alpha=0.4)
            mid_x, mid_y = (query_pos[0] + pos[0])/2, (query_pos[1] + pos[1])/2
            ax.text(mid_x, mid_y, f'{score:.2f}', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

        # Legend
        ax.text(5, 0.5, 'Higher similarity score = More semantically related',
                ha='center', fontsize=11, style='italic')

        plt.tight_layout()

        if save:
            plt.savefig(f'{self.output_dir}/embedding_similarity.png', dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {self.output_dir}/embedding_similarity.png")

        return fig

    def _draw_box(self, ax, x, y, width, height, text, color, alpha=1.0, textsize=10):
        """Helper to draw a colored box with text"""
        box = FancyBboxPatch((x, y), width, height,
                            boxstyle="round,pad=0.05",
                            edgecolor=color,
                            facecolor=color,
                            alpha=alpha,
                            linewidth=2)
        ax.add_patch(box)
        ax.text(x + width/2, y + height/2, text,
               ha='center', va='center', fontsize=textsize,
               color='white' if alpha > 0.5 else 'black',
               fontweight='bold', wrap=True)

    def _draw_arrow(self, ax, x1, y1, x2, y2, style='solid'):
        """Helper to draw an arrow"""
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                               arrowstyle='->,head_width=0.4,head_length=0.4',
                               color=self.colors['dark'],
                               linewidth=2.5,
                               linestyle=style if style == 'dashed' else 'solid')
        ax.add_patch(arrow)

    def generate_all_diagrams(self):
        """Generate all diagrams at once"""
        print("🎨 Generating RAG Visualization Diagrams...\n")

        self.draw_rag_overview()
        plt.close()

        self.draw_indexing_pipeline()
        plt.close()

        self.draw_query_pipeline()
        plt.close()

        self.draw_chunking_strategies()
        plt.close()

        self.draw_embedding_similarity()
        plt.close()

        print("\n✅ All diagrams generated successfully!")
        print(f"📁 Saved to: {self.output_dir}/")


if __name__ == "__main__":
    import os

    # Create diagrams directory if it doesn't exist
    os.makedirs('../diagrams', exist_ok=True)

    # Generate all visualizations
    visualizer = RAGVisualizer(output_dir='../diagrams')
    visualizer.generate_all_diagrams()

    print("\n🎉 Run this script to regenerate diagrams:")
    print("   python utils/visualizations.py")
