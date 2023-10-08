from collections import deque

from pointcloud.host_sync import HostSync


def test_hostsync_init():
    hs = HostSync(streams=["stream1", "stream2"], maxlen=10)
    assert set(hs.queues.keys()) == {"stream1", "stream2"}
    assert isinstance(hs.queues["stream1"], deque)


def test_hostsync_add():
    hs = HostSync(streams=["stream1", "stream2"])
    mock_msg = type("Msg", (object,), {"getSequenceNum": lambda self: 1})()

    hs.add("stream1", mock_msg)
    hs.add("stream2", mock_msg)

    assert hs.queues["stream1"][0]["msg"] == mock_msg
    assert hs.queues["stream1"][0]["seq"] == 1


def test_hostsync_get_matching_seq():
    hs = HostSync(streams=["stream1", "stream2"])
    mock_msg1 = type("Msg", (object,), {"getSequenceNum": lambda self: 1})()
    mock_msg2 = type("Msg", (object,), {"getSequenceNum": lambda self: 2})()

    hs.add("stream1", mock_msg1)
    hs.add("stream1", mock_msg2)
    hs.add("stream2", mock_msg2)

    result = hs.get()

    assert result["stream1"] == mock_msg2
    assert result["stream2"] == mock_msg2


def test_hostsync_get_nonmatching_seq():
    hs = HostSync(streams=["stream1", "stream2"])
    mock_msg1 = type("Msg", (object,), {"getSequenceNum": lambda self: 1})()
    mock_msg2 = type("Msg", (object,), {"getSequenceNum": lambda self: 2})()

    hs.add("stream1", mock_msg1)
    hs.add("stream2", mock_msg2)

    result = hs.get()

    assert result is None


def test_hostsync_get_empty_seq():
    hs = HostSync(streams=["stream1", "stream2"])
    result = hs.get()
    assert result is None
