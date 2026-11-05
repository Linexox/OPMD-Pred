package org.opmd.followup.demo;

import org.junit.Test;
import java.util.Arrays;
import static org.junit.Assert.*;

public class CaptureHintsTest {
    private int[] flat(int value) { int[] values = new int[768]; Arrays.fill(values, value); return values; }
    @Test public void sustainedDarknessAndBrightnessAreReported() {
        CaptureHints hints = new CaptureHints();
        hints.update(flat(0)); hints.update(flat(0));
        assertTrue(hints.update(flat(0)).contains("暗部"));
        hints.update(flat(255)); hints.update(flat(255));
        assertTrue(hints.update(flat(255)).contains("亮部"));
    }
    @Test public void isolatedChangeDoesNotReplaceHint() {
        CaptureHints hints = new CaptureHints();
        hints.update(flat(130)); hints.update(flat(130));
        String steady = hints.update(flat(130));
        assertEquals(steady, hints.update(flat(0)));
        assertEquals(steady, hints.update(flat(130)));
    }
    @Test public void sustainedSpatialChangesAreReported() {
        CaptureHints hints = new CaptureHints();
        int[] first = new int[768], second = new int[768];
        for (int i = 0; i < first.length; i++) { first[i] = i % 2 == 0 ? 80 : 180; second[i] = 260 - first[i]; }
        hints.update(first); hints.update(second); hints.update(first);
        assertTrue(hints.update(second).contains("画面变化"));
    }
    @Test public void exposureChangeAloneDoesNotClaimMovement() {
        CaptureHints hints = new CaptureHints();
        hints.update(flat(70)); hints.update(flat(100)); hints.update(flat(130));
        assertFalse(hints.update(flat(160)).contains("画面变化"));
        hints.reset(); assertEquals("请对准要记录的位置", hints.update(flat(130)));
    }
}
