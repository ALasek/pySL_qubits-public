"""Scrollable Tk window that accumulates matplotlib figures, like Spyder's plot pane."""

import base64
import io
import tkinter as tk


class PlotPane:
    def __init__(self, title="Simulation Plots", width=960, height=780):
        self._root = tk.Tk()
        self._root.title(title)
        self._root.geometry(f"{width}x{height}")
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._alive = True
        self._count = 0

        container = tk.Frame(self._root)
        container.pack(fill=tk.BOTH, expand=True)

        self._canvas = tk.Canvas(container, bg="#f0f0f0", highlightthickness=0)
        vscroll = tk.Scrollbar(container, orient=tk.VERTICAL, command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=vscroll.set)
        vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._inner = tk.Frame(self._canvas, bg="#f0f0f0")
        _win = self._canvas.create_window((0, 0), window=self._inner, anchor="nw")

        self._inner.bind(
            "<Configure>",
            lambda e: self._canvas.configure(scrollregion=self._canvas.bbox("all")),
        )
        self._canvas.bind(
            "<Configure>",
            lambda e: self._canvas.itemconfig(_win, width=e.width),
        )
        self._canvas.bind_all(
            "<MouseWheel>",
            lambda e: self._canvas.yview_scroll(-1 * (e.delta // 120), "units"),
        )

        self._photos = []  # hold references to prevent GC
        self._root.update()

    def _on_close(self):
        self._alive = False
        try:
            self._root.destroy()
        except Exception:
            pass

    def push(self, fig):
        """Render fig as PNG and append it to the pane."""
        if not self._alive:
            return
        self._count += 1

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        buf.seek(0)

        photo = tk.PhotoImage(data=base64.b64encode(buf.getvalue()))
        self._photos.append(photo)

        lbl_sep = tk.Label(
            self._inner,
            text=f"  Plot {self._count}  ",
            bg="#d0d0d0",
            fg="#444444",
            font=("Segoe UI", 8),
            anchor="w",
        )
        lbl_sep.pack(fill=tk.X, padx=0, pady=(6, 0))

        tk.Label(self._inner, image=photo, bg="#f0f0f0", bd=0).pack(
            pady=(2, 0), padx=8, anchor="center"
        )

        try:
            self._root.update()
            self._canvas.yview_moveto(1.0)
        except Exception:
            pass

    def wait(self):
        """Block until the user closes the pane."""
        if self._alive:
            try:
                self._root.mainloop()
            except Exception:
                pass
