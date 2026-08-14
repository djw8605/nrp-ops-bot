"""NRP documentation indexing.

The docs site is Astro Starlight. This package indexes the MDX *source* from
``gitlab.nrp-nautilus.io/prp/nrp-site`` rather than crawling the rendered HTML:
the source keeps heading structure and fenced code blocks intact, and operators
search on exact identifiers that HTML-to-text extraction mangles.
"""
